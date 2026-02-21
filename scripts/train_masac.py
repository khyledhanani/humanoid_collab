"""Train MASAC (Multi-Agent SAC) for the hug task — Stage C.

Both humanoids learn simultaneously:
  - Decentralised stochastic actors (each sees only its own observation).
  - Centralised twin Q-networks (sees joint obs + joint actions — CTDE).
  - Single shared auto-tuned temperature α (cooperative task, same reward).

The Bellman backup for the shared critic includes entropy from **both** agents:
    y = r + γ · (1-d) · (min Q_target(s',a'_h0,a'_h1) − α·(log π_h0 + log π_h1))

Each actor update holds the other agent's actions fixed (detached), so gradients
flow only through the policy being optimised.

Usage:
    # Stage C from scratch
    python scripts/train_masac.py --stage 0 --auto-curriculum

    # Stage C warm-started from Stage A/B checkpoint (both actors initialised)
    python scripts/train_masac.py \\
        --stage 0 --auto-curriculum \\
        --load-actor-from checkpoints/sac_stage_b/final.pt \\
        --save-dir checkpoints/masac_stage_c
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError as exc:
    raise ImportError(
        "MASAC training requires PyTorch. Install with: pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger

AGENTS = ("h0", "h1")

_TARGET_ENTROPY_BY_STAGE: Dict[int, float] = {
    0: -8.0,
    1: -11.0,
    2: -14.0,
    3: -18.0,
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MASAC on HumanoidCollabEnv — Stage C (both agents active)."
    )

    # Environment
    p.add_argument("--task", type=str, default="hug")
    p.add_argument(
        "--physics-profile", type=str, default="train_fast",
        choices=available_physics_profiles(),
    )
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--hold-target", type=int, default=30)

    # MASAC hyperparameters
    p.add_argument("--buffer-size", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--start-steps", type=int, default=20_000)
    p.add_argument("--update-after", type=int, default=20_000)
    p.add_argument("--update-every", type=int, default=2,
                   help="Run gradient steps every N env steps.")
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--actor-hidden-size", type=int, default=512)
    p.add_argument("--critic-hidden-size", type=int, default=1024)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument(
        "--target-entropy-override", type=float, default=None,
        help="Override stage-conditioned target entropy (per-agent).",
    )

    # Warm-start
    p.add_argument(
        "--load-actor-from", type=str, default=None,
        help=(
            "Path to a Stage A/B SAC checkpoint.  "
            "The 'actor' key is loaded into both h0 and h1 actors."
        ),
    )

    # Curriculum
    p.add_argument("--auto-curriculum", action="store_true")
    p.add_argument("--curriculum-window", type=int, default=60)
    p.add_argument("--curriculum-threshold", type=float, default=0.65)
    p.add_argument("--curriculum-min-episodes", type=int, default=80)

    # Checkpointing / logging
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--log-dir", type=str, default="runs/masac_hug_stage_c")
    p.add_argument("--save-dir", type=str, default="checkpoints/masac_hug_stage_c")
    p.add_argument("--save-every-steps", type=int, default=100_000)
    p.add_argument("--print-every-steps", type=int, default=5_000)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="humanoid-collab")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default="masac")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])

    return p.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class SACGaussianActor(nn.Module):
    """Stochastic Gaussian actor with tanh squashing."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Linear(hidden_size, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = self.trunk(obs)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mu, std)
        x_t     = dist.rsample()
        action  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        action, _ = self.forward(obs)
        return action.cpu().numpy()


class MASACCentralCritic(nn.Module):
    """Centralised twin Q-networks for CTDE MASAC.

    Input: concatenation of (obs_h0, obs_h1, act_h0, act_h1).
    """

    def __init__(self, joint_dim: int, hidden_size: int = 1024):
        super().__init__()

        def _mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(joint_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, joint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(joint), self.q2(joint)

    def q_min(self, joint: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(joint)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class MASACReplayBuffer:
    """Circular buffer storing joint transitions from both agents."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        n = int(capacity)
        self._obs_h0      = np.zeros((n, obs_dim), dtype=np.float32)
        self._obs_h1      = np.zeros((n, obs_dim), dtype=np.float32)
        self._act_h0      = np.zeros((n, act_dim), dtype=np.float32)
        self._act_h1      = np.zeros((n, act_dim), dtype=np.float32)
        self._rewards     = np.zeros((n, 1),        dtype=np.float32)
        self._next_obs_h0 = np.zeros((n, obs_dim), dtype=np.float32)
        self._next_obs_h1 = np.zeros((n, obs_dim), dtype=np.float32)
        self._dones       = np.zeros((n, 1),        dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._cap  = n

    def add(
        self,
        obs_h0: np.ndarray, obs_h1: np.ndarray,
        act_h0: np.ndarray, act_h1: np.ndarray,
        reward: float,
        next_obs_h0: np.ndarray, next_obs_h1: np.ndarray,
        done: float,
    ) -> None:
        i = self._ptr
        self._obs_h0[i]      = obs_h0
        self._obs_h1[i]      = obs_h1
        self._act_h0[i]      = act_h0
        self._act_h1[i]      = act_h1
        self._rewards[i]     = reward
        self._next_obs_h0[i] = next_obs_h0
        self._next_obs_h1[i] = next_obs_h1
        self._dones[i]       = done
        self._ptr  = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self._size, size=batch_size)
        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[idx], dtype=torch.float32, device=device)
        return (
            _t(self._obs_h0), _t(self._obs_h1),
            _t(self._act_h0), _t(self._act_h1),
            _t(self._rewards),
            _t(self._next_obs_h0), _t(self._next_obs_h1),
            _t(self._dones),
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# MASAC update
# ---------------------------------------------------------------------------

def masac_update(
    actor_h0: SACGaussianActor,
    actor_h1: SACGaussianActor,
    critic: MASACCentralCritic,
    critic_target: MASACCentralCritic,
    actor_h0_opt: optim.Optimizer,
    actor_h1_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    alpha_opt: optim.Optimizer,
    log_alpha: torch.Tensor,
    target_entropy: float,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
    tau: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    (
        obs_h0, obs_h1,
        act_h0, act_h1,
        rewards,
        next_obs_h0, next_obs_h1,
        dones,
    ) = batch

    alpha = log_alpha.exp().detach()

    def _joint(o0, o1, a0, a1) -> torch.Tensor:
        return torch.cat([o0, o1, a0, a1], dim=-1)

    # ---- Critic update ----
    with torch.no_grad():
        na_h0, nl_h0 = actor_h0(next_obs_h0)
        na_h1, nl_h1 = actor_h1(next_obs_h1)
        q_next = critic_target.q_min(_joint(next_obs_h0, next_obs_h1, na_h0, na_h1))
        # Entropy from both agents enters the cooperative Bellman target
        y = rewards + gamma * (1.0 - dones) * (q_next - alpha * (nl_h0 + nl_h1))

    q1, q2 = critic(_joint(obs_h0, obs_h1, act_h0, act_h1))
    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_opt.step()

    # ---- Actor h0 update (h1 fixed) ----
    new_a_h0, log_pi_h0 = actor_h0(obs_h0)
    with torch.no_grad():
        det_a_h1, _ = actor_h1(obs_h1)
    q_for_h0 = critic.q_min(_joint(obs_h0, obs_h1, new_a_h0, det_a_h1))
    actor_h0_loss = (alpha * log_pi_h0 - q_for_h0).mean()

    actor_h0_opt.zero_grad()
    actor_h0_loss.backward()
    nn.utils.clip_grad_norm_(actor_h0.parameters(), max_grad_norm)
    actor_h0_opt.step()

    # ---- Actor h1 update (h0 fixed) ----
    new_a_h1, log_pi_h1 = actor_h1(obs_h1)
    with torch.no_grad():
        det_a_h0, _ = actor_h0(obs_h0)
    q_for_h1 = critic.q_min(_joint(obs_h0, obs_h1, det_a_h0, new_a_h1))
    actor_h1_loss = (alpha * log_pi_h1 - q_for_h1).mean()

    actor_h1_opt.zero_grad()
    actor_h1_loss.backward()
    nn.utils.clip_grad_norm_(actor_h1.parameters(), max_grad_norm)
    actor_h1_opt.step()

    # ---- Shared temperature update ----
    # Average both agents' log-probs as the temperature gradient signal.
    mean_log_pi = ((log_pi_h0 + log_pi_h1) * 0.5).detach()
    alpha_loss = -(log_alpha * (mean_log_pi + target_entropy)).mean()

    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # ---- Soft target update ----
    with torch.no_grad():
        for p_t, p in zip(critic_target.parameters(), critic.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    return {
        "critic_loss":   critic_loss.item(),
        "actor_h0_loss": actor_h0_loss.item(),
        "actor_h1_loss": actor_h1_loss.item(),
        "alpha_loss":    alpha_loss.item(),
        "alpha":         log_alpha.exp().item(),
        "q_mean":        q1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    actor_h0: SACGaussianActor,
    actor_h1: SACGaussianActor,
    critic: MASACCentralCritic,
    critic_target: MASACCentralCritic,
    actor_h0_opt: optim.Optimizer,
    actor_h1_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    alpha_opt: optim.Optimizer,
    log_alpha: torch.Tensor,
    total_steps: int,
    stage: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "actor_h0":      actor_h0.state_dict(),
            "actor_h1":      actor_h1.state_dict(),
            "critic":        critic.state_dict(),
            "critic_target": critic_target.state_dict(),
            "actor_h0_opt":  actor_h0_opt.state_dict(),
            "actor_h1_opt":  actor_h1_opt.state_dict(),
            "critic_opt":    critic_opt.state_dict(),
            "alpha_opt":     alpha_opt.state_dict(),
            "log_alpha":     log_alpha.detach().cpu(),
            "total_steps":   total_steps,
            "stage":         stage,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    actor_h0: SACGaussianActor,
    actor_h1: SACGaussianActor,
    critic: MASACCentralCritic,
    critic_target: MASACCentralCritic,
    actor_h0_opt: optim.Optimizer,
    actor_h1_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    alpha_opt: optim.Optimizer,
    log_alpha: torch.Tensor,
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    actor_h0.load_state_dict(ckpt["actor_h0"])
    actor_h1.load_state_dict(ckpt["actor_h1"])
    critic.load_state_dict(ckpt["critic"])
    critic_target.load_state_dict(ckpt["critic_target"])
    actor_h0_opt.load_state_dict(ckpt["actor_h0_opt"])
    actor_h1_opt.load_state_dict(ckpt["actor_h1_opt"])
    critic_opt.load_state_dict(ckpt["critic_opt"])
    alpha_opt.load_state_dict(ckpt["alpha_opt"])
    log_alpha.data.copy_(ckpt["log_alpha"].to(device))
    return int(ckpt.get("total_steps", 0)), int(ckpt.get("stage", 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_entropy(stage: int, override: Optional[float], act_dim: int) -> float:
    if override is not None:
        return float(override)
    return _TARGET_ENTROPY_BY_STAGE.get(stage, -float(act_dim))


def _mean_recent(values: List[float], n: int) -> float:
    if not values:
        return 0.0
    k = min(len(values), max(1, n))
    return float(np.mean(values[-k:]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = _resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Environment (both agents fully free, no welding) ----
    env = HumanoidCollabEnv(
        task=args.task,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        hold_target=args.hold_target,
        stage=args.stage,
        physics_profile=args.physics_profile,
        fixed_standing=False,
        control_mode="all",
        weld_h1_only=False,
    )

    obs_dim  = env.observation_space("h0").shape[0]
    act_dim  = env.action_space("h0").shape[0]
    # Centralised critic: obs_h0 + obs_h1 + act_h0 + act_h1
    joint_dim = obs_dim * 2 + act_dim * 2
    stage     = args.stage
    ent       = _target_entropy(stage, args.target_entropy_override, act_dim)

    print(
        f"[train_masac] obs_dim={obs_dim}  act_dim={act_dim}  "
        f"joint_dim={joint_dim}  device={device}"
    )

    # ---- Networks ----
    actor_h0      = SACGaussianActor(obs_dim, act_dim, args.actor_hidden_size).to(device)
    actor_h1      = SACGaussianActor(obs_dim, act_dim, args.actor_hidden_size).to(device)
    critic        = MASACCentralCritic(joint_dim, args.critic_hidden_size).to(device)
    critic_target = MASACCentralCritic(joint_dim, args.critic_hidden_size).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_h0_opt = optim.Adam(actor_h0.parameters(), lr=args.actor_lr)
    actor_h1_opt = optim.Adam(actor_h1.parameters(), lr=args.actor_lr)
    critic_opt   = optim.Adam(critic.parameters(),   lr=args.critic_lr)
    log_alpha    = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_opt    = optim.Adam([log_alpha], lr=args.alpha_lr)

    replay = MASACReplayBuffer(args.buffer_size, obs_dim, act_dim)

    # ---- Warm-start actors from Stage A/B checkpoint ----
    if args.load_actor_from and os.path.isfile(args.load_actor_from):
        ckpt        = torch.load(args.load_actor_from, map_location=device)
        actor_state = ckpt.get("actor")
        if actor_state is not None:
            actor_h0.load_state_dict(actor_state)
            actor_h1.load_state_dict(actor_state)
            print(f"[train_masac] Warm-started both actors from {args.load_actor_from}")
        else:
            print(
                f"[train_masac] WARNING: 'actor' key not found in {args.load_actor_from}. "
                "Starting actors from random init."
            )

    # ---- Resume ----
    global_step = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        global_step, stage = _load_checkpoint(
            args.resume_from,
            actor_h0, actor_h1, critic, critic_target,
            actor_h0_opt, actor_h1_opt, critic_opt, alpha_opt, log_alpha,
            device,
        )
        ent = _target_entropy(stage, args.target_entropy_override, act_dim)
        env.reset(options={"stage": stage})
        print(
            f"[train_masac] Resumed from {args.resume_from} "
            f"at step={global_step}, stage={stage}"
        )

    # ---- Logger ----
    logger = ExperimentLogger.create(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        run_dir=args.log_dir,
        config=vars(args),
    )

    # ---- Tracking ----
    ep_success_hist: List[float] = []
    ep_reward_hist:  List[float] = []
    ep_hold_hist:    List[int]   = []
    ep_steps_in_stage = 0
    n_episodes        = 0
    last_metrics: Dict[str, float] = {}

    # ---- Reset ----
    obs_dict, _ = env.reset()
    obs_h0      = obs_dict["h0"]
    obs_h1      = obs_dict["h1"]
    ep_reward   = 0.0
    ep_max_hold = 0

    t0         = time.time()
    print_step = 0

    # ---- Main loop ----
    for _ in range(args.total_steps):
        global_step += 1

        # Action selection
        if global_step < args.start_steps:
            action_h0 = env.action_space("h0").sample()
            action_h1 = env.action_space("h1").sample()
        else:
            obs_h0_t = torch.as_tensor(obs_h0[None], dtype=torch.float32, device=device)
            obs_h1_t = torch.as_tensor(obs_h1[None], dtype=torch.float32, device=device)
            action_h0 = actor_h0.get_action(obs_h0_t)[0]
            action_h1 = actor_h1.get_action(obs_h1_t)[0]

        obs_new, rew_dict, term_dict, trunc_dict, info_dict = env.step(
            {"h0": action_h0, "h1": action_h1}
        )

        next_obs_h0 = obs_new["h0"]
        next_obs_h1 = obs_new["h1"]
        reward      = float(rew_dict["h0"])   # cooperative — same for both
        terminated  = bool(term_dict["h0"])
        truncated   = bool(trunc_dict["h0"])
        done        = terminated or truncated

        # Store; use terminated not truncated so bootstrapping is correct on timeout
        replay.add(
            obs_h0, obs_h1,
            action_h0, action_h1,
            reward,
            next_obs_h0, next_obs_h1,
            float(terminated),
        )

        obs_h0      = next_obs_h0
        obs_h1      = next_obs_h1
        ep_reward  += reward
        ep_max_hold = max(ep_max_hold, info_dict["h0"].get("hold_steps", 0))

        # ---- MASAC update ----
        if (
            global_step >= args.update_after
            and len(replay) >= args.batch_size
            and global_step % args.update_every == 0
        ):
            for _ in range(args.gradient_steps):
                batch = replay.sample(args.batch_size, device)
                last_metrics = masac_update(
                    actor_h0, actor_h1,
                    critic, critic_target,
                    actor_h0_opt, actor_h1_opt, critic_opt, alpha_opt,
                    log_alpha, ent,
                    batch, args.gamma, args.tau, args.max_grad_norm,
                )

        # ---- Episode end ----
        if done:
            reason  = info_dict["h0"].get("termination_reason", "unknown")
            success = reason == "success"

            ep_success_hist.append(float(success))
            ep_reward_hist.append(ep_reward)
            ep_hold_hist.append(ep_max_hold)
            ep_steps_in_stage += 1
            n_episodes += 1

            # Auto-curriculum
            if args.auto_curriculum and stage < 3:
                if ep_steps_in_stage >= args.curriculum_min_episodes:
                    sr = _mean_recent(ep_success_hist, args.curriculum_window)
                    if sr >= args.curriculum_threshold:
                        stage += 1
                        env.reset(options={"stage": stage})
                        ent = _target_entropy(stage, args.target_entropy_override, act_dim)
                        ep_steps_in_stage = 0
                        print(
                            f"\n[Curriculum] Advanced to stage {stage}  "
                            f"(success_rate={sr:.2f}  step={global_step})\n"
                        )

            logger.log(
                step=global_step,
                common={
                    "episode_reward":  ep_reward,
                    "success":         float(success),
                    "max_hold_steps":  ep_max_hold,
                    "stage":           stage,
                    "n_episodes":      n_episodes,
                },
                algo=last_metrics or {},
            )

            obs_dict, _ = env.reset()
            obs_h0      = obs_dict["h0"]
            obs_h1      = obs_dict["h1"]
            ep_reward   = 0.0
            ep_max_hold = 0

        # ---- Console log ----
        if global_step - print_step >= args.print_every_steps:
            print_step = global_step
            elapsed    = time.time() - t0
            sps        = global_step / max(elapsed, 1e-3)
            sr         = _mean_recent(ep_success_hist, args.curriculum_window)
            mean_r     = _mean_recent(ep_reward_hist, 20)
            mean_hold  = _mean_recent([float(h) for h in ep_hold_hist], 20)
            alpha_val  = log_alpha.exp().item()
            q_mean     = last_metrics.get("q_mean", float("nan"))
            print(
                f"step={global_step:>9,d}  stage={stage}  sps={sps:,.0f}  "
                f"ep={n_episodes:,d}  sr={sr:.3f}  "
                f"mean_r={mean_r:.2f}  hold={mean_hold:.1f}  "
                f"alpha={alpha_val:.4f}  q={q_mean:.3f}"
            )

        # ---- Checkpoint ----
        if global_step % args.save_every_steps == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{global_step:08d}.pt")
            _save_checkpoint(
                ckpt_path,
                actor_h0, actor_h1, critic, critic_target,
                actor_h0_opt, actor_h1_opt, critic_opt, alpha_opt,
                log_alpha, global_step, stage,
            )
            print(f"[ckpt] Saved {ckpt_path}")

    # ---- Final checkpoint ----
    final_path = os.path.join(args.save_dir, "final.pt")
    _save_checkpoint(
        final_path,
        actor_h0, actor_h1, critic, critic_target,
        actor_h0_opt, actor_h1_opt, critic_opt, alpha_opt,
        log_alpha, global_step, stage,
    )
    print(f"\n[train_masac] Done. Final checkpoint: {final_path}")

    env.close()
    logger.finish()


if __name__ == "__main__":
    main()
