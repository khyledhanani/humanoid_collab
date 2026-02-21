"""Train SAC for the staged hug curriculum — Stages A and B.

Stage A: H0 (SAC, full locomotion) vs H1 (torso welded, zero action).
         H0 learns to walk toward a static dummy and initiate a hug.

Stage B: H0 (SAC, full locomotion) vs H1 (lower body welded, scripted arms).
         H0 learns to hug a partner whose arms open reactively on approach.
         Warm-start H0 from a Stage A checkpoint with --resume-from.

Usage:
    # Stage A — train from scratch
    python scripts/train_sac.py \\
        --partner-mode zero --h1-weld-mode torso \\
        --stage 0 --auto-curriculum \\
        --save-dir checkpoints/sac_stage_a

    # Stage B — reactive partner, warm-start from Stage A
    python scripts/train_sac.py \\
        --partner-mode scripted --h1-weld-mode lower_body \\
        --stage 0 --auto-curriculum \\
        --resume-from checkpoints/sac_stage_a/final.pt \\
        --save-dir checkpoints/sac_stage_b
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
        "SAC training requires PyTorch. Install with: pip install -e '.[train]'"
    ) from exc

import mujoco

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.partners.scripted import ScriptedPartner
from humanoid_collab.utils.exp_logging import ExperimentLogger

# ---------------------------------------------------------------------------
# Stage-conditioned target entropy
# Looser early (high exploration), tighter later (stable hold).
# ---------------------------------------------------------------------------
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
        description="Train SAC on HumanoidCollabEnv — Stages A and B."
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

    # H1 / partner configuration
    p.add_argument(
        "--partner-mode", type=str, default="zero",
        choices=["zero", "scripted"],
        help="zero=Stage A (inert dummy), scripted=Stage B (reactive arms)",
    )
    p.add_argument(
        "--h1-weld-mode", type=str, default="torso",
        choices=["torso", "lower_body"],
        help=(
            "torso=Stage A (H1 fully frozen), "
            "lower_body=Stage B (H1 pelvis fixed, arms free)"
        ),
    )

    # SAC hyperparameters
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--start-steps", type=int, default=20_000)
    p.add_argument("--update-after", type=int, default=20_000)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument(
        "--target-entropy-override", type=float, default=None,
        help="Override stage-conditioned target entropy.",
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
    p.add_argument("--log-dir", type=str, default="runs/sac_hug_stage_ab")
    p.add_argument("--save-dir", type=str, default="checkpoints/sac_hug_stage_ab")
    p.add_argument("--save-every-steps", type=int, default=100_000)
    p.add_argument("--print-every-steps", type=int, default=5_000)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume / warm-start from a checkpoint path.")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="humanoid-collab")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default="sac")
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
    """Stochastic Gaussian policy with tanh squashing and reparameterisation."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Linear(hidden_size, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) with tanh correction applied."""
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        # log π(a|s) with tanh-squash correction
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        action, _ = self.forward(obs)
        return action.cpu().numpy()

    @torch.no_grad()
    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        h = self.trunk(obs)
        return torch.tanh(self.mu_head(h)).cpu().numpy()


class SACTwinCritic(nn.Module):
    """Twin Q-networks Q1, Q2 for variance reduction in SAC."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 512):
        super().__init__()
        in_dim = obs_dim + act_dim

        def _mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class SACReplayBuffer:
    """Circular replay buffer for single-agent SAC transitions."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self._obs      = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self._rewards  = np.zeros((self.capacity, 1),       dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._dones    = np.zeros((self.capacity, 1),       dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        i = self._ptr
        self._obs[i]      = obs
        self._actions[i]  = action
        self._rewards[i]  = reward
        self._next_obs[i] = next_obs
        self._dones[i]    = done
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self._size, size=batch_size)
        def _t(arr):
            return torch.as_tensor(arr[idx], dtype=torch.float32, device=device)
        return (
            _t(self._obs),
            _t(self._actions),
            _t(self._rewards),
            _t(self._next_obs),
            _t(self._dones),
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------

def sac_update(
    actor: SACGaussianActor,
    critic: SACTwinCritic,
    critic_target: SACTwinCritic,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    alpha_opt: optim.Optimizer,
    log_alpha: torch.Tensor,
    target_entropy: float,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
    tau: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    obs, actions, rewards, next_obs, dones = batch
    alpha = log_alpha.exp().detach()

    # ---- Critic update ----
    with torch.no_grad():
        next_a, next_log_pi = actor(next_obs)
        q1_next, q2_next = critic_target(next_obs, next_a)
        q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
        y = rewards + gamma * (1.0 - dones) * q_next

    q1, q2 = critic(obs, actions)
    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_opt.step()

    # ---- Actor update ----
    new_a, log_pi = actor(obs)
    q_val = critic.q_min(obs, new_a)
    actor_loss = (alpha * log_pi - q_val).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_opt.step()

    # ---- Temperature update ----
    alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # ---- Soft target update ----
    with torch.no_grad():
        for p_t, p in zip(critic_target.parameters(), critic.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss":  actor_loss.item(),
        "alpha_loss":  alpha_loss.item(),
        "alpha":       log_alpha.exp().item(),
        "q_mean":      q1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_arm_actuator_positions(env: HumanoidCollabEnv, partner: str = "h1") -> List[int]:
    """Return positions within partner's action vector that are arm actuators."""
    arm_tokens = ("shoulder", "elbow")
    positions: List[int] = []
    for pos, full_idx in enumerate(env._control_actuator_idx[partner]):
        act_name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(full_idx)
        )
        if act_name and any(t in act_name for t in arm_tokens):
            positions.append(pos)
    return positions


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
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    actor: SACGaussianActor,
    critic: SACTwinCritic,
    critic_target: SACTwinCritic,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    alpha_opt: optim.Optimizer,
    log_alpha: torch.Tensor,
    total_steps: int,
    stage: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "actor":         actor.state_dict(),
            "critic":        critic.state_dict(),
            "critic_target": critic_target.state_dict(),
            "actor_opt":     actor_opt.state_dict(),
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
    actor: SACGaussianActor,
    critic: SACTwinCritic,
    critic_target: SACTwinCritic,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    alpha_opt: optim.Optimizer,
    log_alpha: torch.Tensor,
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    critic_target.load_state_dict(ckpt["critic_target"])
    actor_opt.load_state_dict(ckpt["actor_opt"])
    critic_opt.load_state_dict(ckpt["critic_opt"])
    alpha_opt.load_state_dict(ckpt["alpha_opt"])
    log_alpha.data.copy_(ckpt["log_alpha"].to(device))
    return int(ckpt.get("total_steps", 0)), int(ckpt.get("stage", 0))


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

    # ---- Environment ----
    # H1 is always welded in Stage A/B.  H1 weld mode selects torso (Stage A)
    # or lower_body (Stage B, arms free for scripted controller).
    env = HumanoidCollabEnv(
        task=args.task,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        hold_target=args.hold_target,
        stage=args.stage,
        physics_profile=args.physics_profile,
        fixed_standing=False,
        control_mode="all",
        weld_h1_only=True,
        h1_weld_mode=args.h1_weld_mode,
    )

    obs_dim = env.observation_space("h0").shape[0]
    act_dim = env.action_space("h0").shape[0]
    stage   = args.stage
    ent     = _target_entropy(stage, args.target_entropy_override, act_dim)

    print(
        f"[train_sac] obs_dim={obs_dim}  act_dim={act_dim}  "
        f"device={device}  partner_mode={args.partner_mode}  "
        f"h1_weld_mode={args.h1_weld_mode}"
    )

    # ---- Scripted partner (Stage B only) ----
    scripted_partner: Optional[ScriptedPartner] = None
    if args.partner_mode == "scripted":
        arm_positions = _build_arm_actuator_positions(env, partner="h1")
        scripted_partner = ScriptedPartner(
            id_cache=env.id_cache,
            arm_actuator_positions=arm_positions,
            act_dim=act_dim,
        )
        print(f"[train_sac] Scripted partner arm positions: {arm_positions}")

    # ---- Networks ----
    actor         = SACGaussianActor(obs_dim, act_dim, args.hidden_size).to(device)
    critic        = SACTwinCritic(obs_dim, act_dim, args.hidden_size).to(device)
    critic_target = SACTwinCritic(obs_dim, act_dim, args.hidden_size).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_opt  = optim.Adam(actor.parameters(),  lr=args.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)
    log_alpha  = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_opt  = optim.Adam([log_alpha], lr=args.alpha_lr)

    replay = SACReplayBuffer(args.buffer_size, obs_dim, act_dim)

    # ---- Resume ----
    global_step = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        global_step, stage = _load_checkpoint(
            args.resume_from,
            actor, critic, critic_target,
            actor_opt, critic_opt, alpha_opt, log_alpha,
            device,
        )
        ent = _target_entropy(stage, args.target_entropy_override, act_dim)
        env.reset(options={"stage": stage})
        print(
            f"[train_sac] Resumed from {args.resume_from} "
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

    # ---- Curriculum / episode tracking ----
    ep_success_hist: List[float] = []
    ep_reward_hist:  List[float] = []
    ep_hold_hist:    List[int]   = []
    ep_steps_in_stage = 0
    n_episodes = 0
    last_metrics: Dict[str, float] = {}

    # ---- Reset ----
    obs_dict, _ = env.reset()
    obs_h0  = obs_dict["h0"]
    ep_reward   = 0.0
    ep_max_hold = 0

    if scripted_partner is not None:
        scripted_partner.reset()

    t0         = time.time()
    print_step = 0

    # ---- Main loop ----
    for _ in range(args.total_steps):
        global_step += 1

        # Action: random during warm-up, SAC thereafter
        if global_step < args.start_steps:
            action_h0 = env.action_space("h0").sample()
        else:
            obs_t = torch.as_tensor(obs_h0[None], dtype=torch.float32, device=device)
            action_h0 = actor.get_action(obs_t)[0]

        # H1 action: zero (Stage A) or scripted (Stage B)
        if scripted_partner is not None:
            action_h1 = scripted_partner.get_action(env.data)
        else:
            action_h1 = np.zeros(act_dim, dtype=np.float32)

        obs_new, rew_dict, term_dict, trunc_dict, info_dict = env.step(
            {"h0": action_h0, "h1": action_h1}
        )

        next_obs_h0 = obs_new["h0"]
        reward      = float(rew_dict["h0"])
        terminated  = bool(term_dict["h0"])
        truncated   = bool(trunc_dict["h0"])
        done        = terminated or truncated

        # Store transition.  Use terminated (not truncated) as the done signal
        # so the value bootstrap is not cut on timeout.
        replay.add(obs_h0, action_h0, reward, next_obs_h0, float(terminated))

        obs_h0       = next_obs_h0
        ep_reward   += reward
        ep_max_hold  = max(ep_max_hold, info_dict["h0"].get("hold_steps", 0))

        # ---- SAC update ----
        if (
            global_step >= args.update_after
            and len(replay) >= args.batch_size
            and global_step % args.update_every == 0
        ):
            for _ in range(args.gradient_steps):
                batch = replay.sample(args.batch_size, device)
                last_metrics = sac_update(
                    actor, critic, critic_target,
                    actor_opt, critic_opt, alpha_opt,
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
            ep_reward   = 0.0
            ep_max_hold = 0

            if scripted_partner is not None:
                scripted_partner.reset()

        # ---- Periodic console log ----
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
                actor, critic, critic_target,
                actor_opt, critic_opt, alpha_opt, log_alpha,
                global_step, stage,
            )
            print(f"[ckpt] Saved {ckpt_path}")

    # ---- Final checkpoint ----
    final_path = os.path.join(args.save_dir, "final.pt")
    _save_checkpoint(
        final_path,
        actor, critic, critic_target,
        actor_opt, critic_opt, alpha_opt, log_alpha,
        global_step, stage,
    )
    print(f"\n[train_sac] Done. Final checkpoint: {final_path}")

    env.close()
    logger.finish()


if __name__ == "__main__":
    main()
