"""Train IPPO policies for HumanoidCollabEnv.

Default settings target fixed-standing, arms-only handshake training.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.normal import Normal
except ImportError as exc:
    raise ImportError(
        "IPPO training requires PyTorch. Install training dependencies with: "
        "pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger


AGENTS = ("h0", "h1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IPPO on HumanoidCollabEnv.")

    parser.add_argument("--task", type=str, default="handshake", choices=["hug", "handshake", "box_lift"])
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--physics-profile", type=str, default="default", choices=available_physics_profiles())
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=30)
    parser.add_argument("--fixed-standing", action="store_true", default=True)
    parser.add_argument("--control-mode", type=str, default="arms_only", choices=["all", "arms_only"])

    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true")

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=4)

    parser.add_argument("--log-dir", type=str, default="runs/ippo_handshake_fixed_arms")
    parser.add_argument("--save-dir", type=str, default="checkpoints/ippo_handshake_fixed_arms")
    parser.add_argument("--save-every-updates", type=int, default=25)
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a checkpoint path.")
    parser.add_argument("--print-every-updates", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="ippo")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--auto-curriculum", action="store_true", help="Advance curriculum stage from success rate.")
    parser.add_argument(
        "--curriculum-window-episodes",
        type=int,
        default=40,
        help="Episode window used to evaluate promotion success rate.",
    )
    parser.add_argument(
        "--curriculum-success-threshold",
        type=float,
        default=0.6,
        help="Default success-rate threshold for stage promotion.",
    )
    parser.add_argument(
        "--curriculum-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional per-promotion thresholds (from current stage upward).",
    )
    parser.add_argument(
        "--curriculum-min-updates-per-stage",
        type=int,
        default=25,
        help="Minimum number of updates before promoting from a stage.",
    )

    parser.add_argument("--fast-learn", action="store_true", help="Enable more aggressive PPO update settings.")
    parser.add_argument(
        "--adaptive-clip-by-kl",
        action="store_true",
        help="Adapt PPO clip coefficient based on measured KL.",
    )
    parser.add_argument(
        "--adaptive-lr-by-kl",
        action="store_true",
        help="Adapt optimizer learning rate based on measured KL.",
    )
    parser.add_argument("--target-kl", type=float, default=0.03, help="KL target for adaptive policies.")
    parser.add_argument("--clip-min", type=float, default=0.12)
    parser.add_argument("--clip-max", type=float, default=0.40)
    parser.add_argument("--clip-scale-up", type=float, default=1.10)
    parser.add_argument("--clip-scale-down", type=float, default=0.85)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--lr-scale-up", type=float, default=1.08)
    parser.add_argument("--lr-scale-down", type=float, default=0.80)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_size, act_dim)
        self.v = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mu = self.mu(x)
        v = self.v(x).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        log_prob = log_prob - correction
        return action, log_prob, v

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        pre_tanh = torch.atanh(clipped_actions)
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        correction = torch.log(1.0 - clipped_actions.pow(2) + 1e-6).sum(-1)
        log_prob = log_prob - correction
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, v


@dataclass
class Buffer:
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


def make_buffer(rollout_steps: int, obs_dim: int, act_dim: int) -> Buffer:
    return Buffer(
        obs=np.zeros((rollout_steps, obs_dim), dtype=np.float32),
        actions=np.zeros((rollout_steps, act_dim), dtype=np.float32),
        logp=np.zeros((rollout_steps,), dtype=np.float32),
        rewards=np.zeros((rollout_steps,), dtype=np.float32),
        dones=np.zeros((rollout_steps,), dtype=np.float32),
        values=np.zeros((rollout_steps,), dtype=np.float32),
        advantages=np.zeros((rollout_steps,), dtype=np.float32),
        returns=np.zeros((rollout_steps,), dtype=np.float32),
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    last_done: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - last_done
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def make_env(args: argparse.Namespace):
    kwargs = dict(
        task=args.task,
        stage=args.stage,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        hold_target=args.hold_target,
        physics_profile=args.physics_profile,
        fixed_standing=args.fixed_standing,
        control_mode=args.control_mode,
    )
    return HumanoidCollabEnv(**kwargs)


def _reset_env(env, seed: Optional[int], stage: int):
    return env.reset(seed=seed, options={"stage": int(stage)})


def _resolve_curriculum_threshold(
    current_stage: int,
    start_stage: int,
    thresholds: Optional[list[float]],
    default_threshold: float,
) -> float:
    if not thresholds:
        return float(default_threshold)
    idx = current_stage - start_stage
    if idx < 0:
        idx = 0
    if idx >= len(thresholds):
        return float(thresholds[-1])
    return float(thresholds[idx])


def train(args: argparse.Namespace) -> None:
    if args.fast_learn:
        args.adaptive_clip_by_kl = True
        args.adaptive_lr_by_kl = True
        args.clip_coef = max(args.clip_coef, 0.28)
        args.clip_max = max(args.clip_max, 0.45)
        args.lr = max(args.lr, 2e-4)
        args.ppo_epochs = max(args.ppo_epochs, 6)
        args.target_kl = max(args.target_kl, 0.03)

    if args.anneal_lr and args.adaptive_lr_by_kl:
        raise ValueError("--anneal-lr and --adaptive-lr-by-kl cannot be enabled together.")
    if args.curriculum_window_episodes <= 0:
        raise ValueError("--curriculum-window-episodes must be > 0.")
    if args.curriculum_min_updates_per_stage <= 0:
        raise ValueError("--curriculum-min-updates-per-stage must be > 0.")
    if args.clip_min <= 0 or args.clip_min > args.clip_max:
        raise ValueError("Invalid clip bounds: require 0 < clip-min <= clip-max.")
    if args.lr_min <= 0 or args.lr_min > args.lr_max:
        raise ValueError("Invalid lr bounds: require 0 < lr-min <= lr-max.")
    if args.target_kl <= 0:
        raise ValueError("--target-kl must be > 0.")
    if args.curriculum_thresholds is not None:
        for x in args.curriculum_thresholds:
            if x < 0.0 or x > 1.0:
                raise ValueError("--curriculum-thresholds values must be in [0, 1].")

    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    env = make_env(args)
    max_stage = int(env.task_config.num_curriculum_stages) - 1
    if max_stage < 0:
        max_stage = 0
    current_stage = int(max(0, min(args.stage, max_stage)))
    if (
        args.task == "handshake"
        and bool(args.fixed_standing)
        and args.control_mode == "arms_only"
        and current_stage == 0
        and max_stage >= 2
    ):
        current_stage = 2
        print(
            "curriculum guard: using stage 2 instead of stage 0 for "
            "handshake + fixed-standing + arms_only."
        )

    obs, _ = _reset_env(env, seed=args.seed, stage=current_stage)

    obs_dim = int(env.observation_space("h0").shape[0])
    act_dim = int(env.action_space("h0").shape[0])
    if args.minibatch_size > args.rollout_steps:
        raise ValueError("minibatch-size must be <= rollout-steps.")
    if args.rollout_steps % args.minibatch_size != 0:
        raise ValueError("rollout-steps must be divisible by minibatch-size.")

    policies: Dict[str, ActorCritic] = {
        agent: ActorCritic(obs_dim, act_dim, args.hidden_size).to(device) for agent in AGENTS
    }
    optimizers: Dict[str, optim.Optimizer] = {
        agent: optim.Adam(policies[agent].parameters(), lr=args.lr, eps=1e-5) for agent in AGENTS
    }

    os.makedirs(args.log_dir, exist_ok=True)
    wandb_enabled = (not args.no_wandb) and args.wandb_mode != "disabled"
    logger = ExperimentLogger.create(
        enabled=wandb_enabled,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        run_dir=args.log_dir,
        config=vars(args),
    )
    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    updates = max(1, args.total_steps // args.rollout_steps)
    start_update = 1
    ep_return = 0.0
    ep_len = 0
    completed_returns = []
    completed_lengths = []
    completed_successes = []
    start_time = time.time()
    updates_in_stage = 0
    current_clip_coef = float(args.clip_coef)
    current_lr = float(args.lr)
    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"--resume-from checkpoint not found: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)

        ckpt_obs_dim = int(ckpt.get("obs_dim", obs_dim))
        ckpt_act_dim = int(ckpt.get("act_dim", act_dim))
        if ckpt_obs_dim != obs_dim or ckpt_act_dim != act_dim:
            raise ValueError(
                "Resume checkpoint dimensions do not match current env config: "
                f"ckpt(obs={ckpt_obs_dim}, act={ckpt_act_dim}) "
                f"!= env(obs={obs_dim}, act={act_dim})."
            )
        if "policies" not in ckpt:
            raise KeyError(f"Checkpoint at '{args.resume_from}' has no 'policies' key.")

        for agent in AGENTS:
            policies[agent].load_state_dict(ckpt["policies"][agent])
        if "optimizers" in ckpt:
            for agent in AGENTS:
                if agent in ckpt["optimizers"]:
                    optimizers[agent].load_state_dict(ckpt["optimizers"][agent])

        global_step = int(ckpt.get("global_step", 0))
        resume_update = int(ckpt.get("update", global_step // max(1, args.rollout_steps)))
        start_update = max(1, resume_update + 1)
        updates_in_stage = max(0, int(ckpt.get("updates_in_stage", 0)))
        current_stage = int(np.clip(int(ckpt.get("stage", current_stage)), 0, max_stage))
        current_clip_coef = float(ckpt.get("clip_coef", current_clip_coef))
        current_lr = float(ckpt.get("lr", current_lr))
        obs, _ = _reset_env(env, seed=None, stage=current_stage)
        print(
            f"resumed from {args.resume_from}: "
            f"global_step={global_step} update={resume_update} stage={current_stage}"
        )

    for update in range(start_update, updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / updates
            lr_now = args.lr * frac
            current_lr = float(np.clip(lr_now, args.lr_min, args.lr_max))
        for opt in optimizers.values():
            for g in opt.param_groups:
                g["lr"] = current_lr

        buffers = {agent: make_buffer(args.rollout_steps, obs_dim, act_dim) for agent in AGENTS}
        last_done = 0.0

        for t in range(args.rollout_steps):
            action_dict = {}
            for agent in AGENTS:
                obs_t = torch.as_tensor(obs[agent], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, logp_t, value_t = policies[agent].act(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
                action_dict[agent] = action_np

                buffers[agent].obs[t] = obs[agent]
                buffers[agent].actions[t] = action_np
                buffers[agent].logp[t] = float(logp_t.item())
                buffers[agent].values[t] = float(value_t.item())

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            done = bool(terminations["h0"] or truncations["h0"])
            done_f = 1.0 if done else 0.0

            for agent in AGENTS:
                buffers[agent].rewards[t] = float(rewards[agent])
                buffers[agent].dones[t] = done_f

            ep_return += float(rewards["h0"])
            ep_len += 1
            global_step += 1
            last_done = done_f

            if done:
                term_reason = str(infos["h0"].get("termination_reason", "unknown"))
                completed_returns.append(ep_return)
                completed_lengths.append(ep_len)
                completed_successes.append(1.0 if term_reason == "success" else 0.0)
                ep_return = 0.0
                ep_len = 0
                obs, _ = _reset_env(env, seed=None, stage=current_stage)
            else:
                obs = next_obs

        for agent in AGENTS:
            with torch.no_grad():
                last_obs_t = torch.as_tensor(obs[agent], dtype=torch.float32, device=device).unsqueeze(0)
                _, _, last_value_t = policies[agent].act(last_obs_t)
                last_value = float(last_value_t.item())

            adv, ret = compute_gae(
                rewards=buffers[agent].rewards,
                values=buffers[agent].values,
                dones=buffers[agent].dones,
                last_value=last_value,
                last_done=last_done,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            buffers[agent].advantages = adv
            buffers[agent].returns = ret

        metrics = {}
        for agent in AGENTS:
            b = buffers[agent]
            b_obs = torch.as_tensor(b.obs, dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(b.actions, dtype=torch.float32, device=device)
            b_logp_old = torch.as_tensor(b.logp, dtype=torch.float32, device=device)
            b_adv = torch.as_tensor(b.advantages, dtype=torch.float32, device=device)
            b_returns = torch.as_tensor(b.returns, dtype=torch.float32, device=device)

            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            pg_losses = []
            vf_losses = []
            ent_losses = []
            approx_kls = []
            clip_fracs = []

            idxs = np.arange(args.rollout_steps)
            for _ in range(args.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, args.rollout_steps, args.minibatch_size):
                    mb_idx = idxs[start : start + args.minibatch_size]
                    mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

                    new_logp, entropy, new_values = policies[agent].evaluate_actions(
                        b_obs[mb_idx_t], b_actions[mb_idx_t]
                    )
                    log_ratio = new_logp - b_logp_old[mb_idx_t]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_frac = (((ratio - 1.0).abs() > current_clip_coef).float().mean())

                    mb_adv = b_adv[mb_idx_t]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - current_clip_coef, 1.0 + current_clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    mb_returns = b_returns[mb_idx_t]
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = pg_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                    optimizers[agent].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policies[agent].parameters(), args.max_grad_norm)
                    optimizers[agent].step()

                    pg_losses.append(float(pg_loss.item()))
                    vf_losses.append(float(value_loss.item()))
                    ent_losses.append(float(entropy_loss.item()))
                    approx_kls.append(float(approx_kl.item()))
                    clip_fracs.append(float(clip_frac.item()))

            explained_var = float(
                1.0
                - np.var(b.returns - b.values) / (np.var(b.returns) + 1e-8)
            )

            metrics[agent] = {
                "pg_loss": float(np.mean(pg_losses)),
                "vf_loss": float(np.mean(vf_losses)),
                "entropy": float(np.mean(ent_losses)),
                "approx_kl": float(np.mean(approx_kls)),
                "clip_frac": float(np.mean(clip_fracs)),
                "explained_var": explained_var,
            }

        sps = int(global_step / max(1e-6, (time.time() - start_time)))
        mean_return = float(np.mean(completed_returns[-20:])) if completed_returns else 0.0
        mean_ep_len = float(np.mean(completed_lengths[-20:])) if completed_lengths else 0.0
        mean_success_20 = float(np.mean(completed_successes[-20:])) if completed_successes else 0.0
        success_window = float(
            np.mean(completed_successes[-args.curriculum_window_episodes :])
        ) if completed_successes else 0.0
        mean_kl = float(np.mean([metrics["h0"]["approx_kl"], metrics["h1"]["approx_kl"]]))
        updates_in_stage += 1

        if args.adaptive_clip_by_kl:
            if mean_kl < 0.5 * args.target_kl:
                current_clip_coef = float(np.clip(
                    current_clip_coef * args.clip_scale_up, args.clip_min, args.clip_max
                ))
            elif mean_kl > args.target_kl:
                current_clip_coef = float(np.clip(
                    current_clip_coef * args.clip_scale_down, args.clip_min, args.clip_max
                ))

        if args.adaptive_lr_by_kl:
            if mean_kl < 0.5 * args.target_kl:
                current_lr = float(np.clip(current_lr * args.lr_scale_up, args.lr_min, args.lr_max))
            elif mean_kl > args.target_kl:
                current_lr = float(np.clip(current_lr * args.lr_scale_down, args.lr_min, args.lr_max))

        stage_up_msg = None
        if args.auto_curriculum and current_stage < max_stage:
            threshold = _resolve_curriculum_threshold(
                current_stage=current_stage,
                start_stage=int(args.stage),
                thresholds=args.curriculum_thresholds,
                default_threshold=args.curriculum_success_threshold,
            )
            enough_episodes = len(completed_successes) >= args.curriculum_window_episodes
            enough_updates = updates_in_stage >= args.curriculum_min_updates_per_stage
            if enough_episodes and enough_updates and success_window >= threshold:
                prev_stage = current_stage
                current_stage += 1
                updates_in_stage = 0
                stage_up_msg = (
                    f"curriculum: stage {prev_stage} -> {current_stage} "
                    f"(success_window={success_window:.2f} threshold={threshold:.2f})"
                )

        if update % args.print_every_updates == 0:
            print(
                f"update={update}/{updates} step={global_step} sps={sps} "
                f"stage={current_stage} ret20={mean_return:.2f} len20={mean_ep_len:.1f} "
                f"succ20={mean_success_20:.2f} succW={success_window:.2f} "
                f"h0_kl={metrics['h0']['approx_kl']:.4f} h1_kl={metrics['h1']['approx_kl']:.4f} "
                f"clip={current_clip_coef:.3f} lr={current_lr:.6f}"
            )
            if stage_up_msg is not None:
                print(stage_up_msg)

        algo_metrics = {
            "mean_kl": mean_kl,
            "clip_coef": current_clip_coef,
            "lr": current_lr,
        }
        for agent in AGENTS:
            for k, v in metrics[agent].items():
                algo_metrics[f"{agent}/{k}"] = v
        logger.log(
            step=global_step,
            common={
                "sps": sps,
                "ep_return_mean_20": mean_return,
                "ep_len_mean_20": mean_ep_len,
                "ep_success_mean_20": mean_success_20,
                "ep_success_window": success_window,
                "curriculum_stage": current_stage,
            },
            algo=algo_metrics,
            task={
                "stage": current_stage,
            },
        )

        if update % args.save_every_updates == 0 or update == updates:
            ckpt_path = os.path.join(args.save_dir, f"ippo_update_{update:06d}.pt")
            payload = {
                "algo": "ippo",
                "args": vars(args),
                "global_step": global_step,
                "update": update,
                "stage": current_stage,
                "updates_in_stage": updates_in_stage,
                "clip_coef": current_clip_coef,
                "lr": current_lr,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "policies": {a: policies[a].state_dict() for a in AGENTS},
                "optimizers": {a: optimizers[a].state_dict() for a in AGENTS},
            }
            torch.save(payload, ckpt_path)

    env.close()
    logger.finish()
    print("Training complete.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
