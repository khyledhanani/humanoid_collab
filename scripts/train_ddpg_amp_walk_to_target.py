#!/usr/bin/env python3
"""Train single-agent DDPG on walk_to_target with AMP style reward blending."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise ImportError(
        "DDPG training requires PyTorch. Install training dependencies with: "
        "pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.amp.amp_obs import AMPObsBuilder
from humanoid_collab.amp.discriminator import AMPDiscriminator, AMPDiscriminatorTrainer
from humanoid_collab.amp.motion_buffer import MotionReplayBuffer, PolicyTransitionBuffer
from humanoid_collab.amp.motion_data import load_motion_dataset
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train single-agent DDPG+AMP on walk_to_target."
    )

    parser.add_argument("--task", type=str, default="walk_to_target", choices=["walk_to_target"])
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--physics-profile", type=str, default="default", choices=available_physics_profiles())
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=20)
    parser.add_argument("--fixed-standing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control-mode", type=str, default="all", choices=["all", "arms_only"])
    parser.add_argument("--observation-mode", type=str, default="proprio", choices=["proprio"])

    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=2_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--action-l2", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    parser.add_argument("--exploration-noise-start", type=float, default=0.25)
    parser.add_argument("--exploration-noise-end", type=float, default=0.05)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=4)

    parser.add_argument("--amp-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-motion-data-dir", type=str, default=None)
    parser.add_argument(
        "--amp-motion-categories",
        type=str,
        nargs="+",
        default=["standing", "walking", "reaching"],
    )
    parser.add_argument("--amp-disc-checkpoint", type=str, default=None)
    parser.add_argument("--amp-disc-hidden-sizes", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--amp-disc-lr", type=float, default=1e-4)
    parser.add_argument("--amp-disc-batch-size", type=int, default=1024)
    parser.add_argument("--amp-disc-updates", type=int, default=5)
    parser.add_argument("--amp-lambda-gp", type=float, default=10.0)
    parser.add_argument("--amp-warmup-steps", type=int, default=10_000)
    parser.add_argument("--amp-policy-buffer-size", type=int, default=500_000)
    parser.add_argument("--amp-freeze-discriminator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-reward-scale", type=float, default=2.0)
    parser.add_argument("--amp-clip-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-include-root-height", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-normalize-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-reward-anchor-momentum", type=float, default=0.05)
    parser.add_argument("--amp-weight-override", type=float, default=None)
    parser.add_argument("--amp-weight-stage0", type=float, default=0.45)
    parser.add_argument("--amp-weight-stage1", type=float, default=0.40)
    parser.add_argument("--amp-weight-stage2", type=float, default=0.35)
    parser.add_argument("--amp-weight-stage3", type=float, default=0.30)

    parser.add_argument("--auto-curriculum", action="store_true")
    parser.add_argument("--curriculum-window-episodes", type=int, default=80)
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.70)
    parser.add_argument("--curriculum-thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--curriculum-min-episodes-per-stage", type=int, default=120)

    parser.add_argument("--log-dir", type=str, default="runs/ddpg_amp_walk_to_target")
    parser.add_argument("--save-dir", type=str, default="checkpoints/ddpg_amp_walk_to_target")
    parser.add_argument("--save-every-steps", type=int, default=50_000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--print-every-steps", type=int, default=2_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="ddpg-amp-walk")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
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


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        if capacity <= 0:
            raise ValueError("Replay capacity must be > 0")
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = float(reward)
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        )


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau)
            p_t.data.add_(tau * p.data)


def _amp_weight_for_stage(args: argparse.Namespace, stage: int) -> float:
    if args.amp_weight_override is not None:
        return float(args.amp_weight_override)
    by_stage = {
        0: float(args.amp_weight_stage0),
        1: float(args.amp_weight_stage1),
        2: float(args.amp_weight_stage2),
        3: float(args.amp_weight_stage3),
    }
    return float(by_stage.get(int(stage), args.amp_weight_stage3))


def _resolve_curriculum_threshold(
    current_stage: int,
    start_stage: int,
    thresholds: Optional[List[float]],
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


def _update_running_anchor(current: Optional[float], new_value: float, momentum: float) -> float:
    if current is None:
        return float(new_value)
    m = float(np.clip(momentum, 1e-4, 1.0))
    return float((1.0 - m) * current + m * new_value)


def _normalized_amp_reward(
    d_score: np.ndarray,
    d_real_anchor: Optional[float],
    d_fake_anchor: Optional[float],
    reward_scale: float,
    clip_reward: bool,
) -> np.ndarray:
    if d_real_anchor is None or d_fake_anchor is None:
        return np.zeros_like(d_score, dtype=np.float32)
    denom = float(max(1e-6, abs(d_real_anchor - d_fake_anchor)))
    reward = (d_score - d_fake_anchor) / denom
    if clip_reward:
        reward = np.clip(reward, 0.0, 1.0)
    return (float(reward_scale) * reward).astype(np.float32)


def _noise_std(args: argparse.Namespace, step_count: int) -> float:
    if step_count <= args.start_steps:
        return float(args.exploration_noise_start)
    denom = max(1, args.total_steps - args.start_steps)
    frac = min(1.0, max(0.0, (step_count - args.start_steps) / denom))
    return float(
        args.exploration_noise_start
        + frac * (args.exploration_noise_end - args.exploration_noise_start)
    )


def _build_env(args: argparse.Namespace, stage: int, seed: Optional[int]):
    env = HumanoidCollabEnv(
        task=args.task,
        stage=int(stage),
        horizon=int(args.horizon),
        frame_skip=int(args.frame_skip),
        hold_target=int(args.hold_target),
        physics_profile=args.physics_profile,
        fixed_standing=bool(args.fixed_standing),
        control_mode=args.control_mode,
        observation_mode=args.observation_mode,
    )
    obs, infos = env.reset(seed=seed, options={"stage": int(stage)})
    return env, obs, infos


def train(args: argparse.Namespace) -> None:
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0")
    if args.buffer_size <= args.batch_size:
        raise ValueError("--buffer-size should be larger than --batch-size")
    if args.curriculum_window_episodes <= 0:
        raise ValueError("--curriculum-window-episodes must be > 0")
    if args.curriculum_min_episodes_per_stage <= 0:
        raise ValueError("--curriculum-min-episodes-per-stage must be > 0")
    if args.exploration_noise_start < 0.0 or args.exploration_noise_end < 0.0:
        raise ValueError("Exploration noise values must be non-negative")
    if args.fixed_standing:
        raise ValueError("walk_to_target requires locomotion; use --no-fixed-standing.")
    if args.control_mode != "all":
        raise ValueError("walk_to_target is intended for full-body locomotion; use --control-mode all.")
    if args.amp_enable and not args.amp_motion_data_dir:
        raise ValueError("--amp-enable requires --amp-motion-data-dir.")
    if args.amp_enable and args.amp_freeze_discriminator and args.amp_disc_checkpoint is None:
        raise ValueError(
            "--amp-freeze-discriminator requires --amp-disc-checkpoint so AMP reward is meaningful."
        )

    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    probe_env = HumanoidCollabEnv(
        task=args.task,
        stage=int(args.stage),
        horizon=int(args.horizon),
        frame_skip=int(args.frame_skip),
        hold_target=int(args.hold_target),
        physics_profile=args.physics_profile,
        fixed_standing=False,
        control_mode="all",
        observation_mode="proprio",
    )
    max_stage = max(0, int(probe_env.task_config.num_curriculum_stages) - 1)
    probe_env.close()

    current_stage = int(np.clip(int(args.stage), 0, max_stage))
    env, obs_dict, _ = _build_env(args, stage=current_stage, seed=args.seed)
    obs = np.asarray(obs_dict["h0"], dtype=np.float32)
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space("h0").shape[0])

    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    critic = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    actor_target = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    critic_target = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    _hard_update(actor_target, actor)
    _hard_update(critic_target, critic)

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(capacity=args.buffer_size, obs_dim=obs_dim, act_dim=act_dim)

    amp_motion_buffer: Optional[MotionReplayBuffer] = None
    amp_policy_buffer: Optional[PolicyTransitionBuffer] = None
    amp_discriminator: Optional[AMPDiscriminator] = None
    amp_disc_trainer: Optional[AMPDiscriminatorTrainer] = None
    amp_obs_builder: Optional[AMPObsBuilder] = None
    amp_obs_dim = 0
    amp_d_real_anchor: Optional[float] = None
    amp_d_fake_anchor: Optional[float] = None

    if args.amp_enable:
        motion_dataset = load_motion_dataset(
            args.amp_motion_data_dir,
            categories=tuple(args.amp_motion_categories),
        )
        if motion_dataset.num_clips <= 0:
            env.close()
            raise RuntimeError("No motion clips loaded for AMP.")

        amp_obs_builder = AMPObsBuilder(
            id_cache=env.id_cache,
            include_root_height=bool(args.amp_include_root_height),
            include_root_orientation=True,
            include_joint_positions=True,
            include_joint_velocities=True,
        )
        amp_motion_buffer = MotionReplayBuffer(motion_dataset, amp_obs_builder)
        amp_obs_dim = int(amp_motion_buffer.obs_dim)
        amp_policy_buffer = PolicyTransitionBuffer(
            capacity=int(args.amp_policy_buffer_size),
            obs_dim=amp_obs_dim,
        )
        amp_discriminator = AMPDiscriminator(
            obs_dim=amp_obs_dim,
            hidden_sizes=tuple(args.amp_disc_hidden_sizes),
        ).to(device)
        if not args.amp_freeze_discriminator:
            amp_disc_trainer = AMPDiscriminatorTrainer(
                discriminator=amp_discriminator,
                lr=float(args.amp_disc_lr),
                lambda_gp=float(args.amp_lambda_gp),
                n_updates=int(args.amp_disc_updates),
                device=str(device),
            )
        if args.amp_disc_checkpoint:
            amp_ckpt = torch.load(args.amp_disc_checkpoint, map_location=device)
            amp_state = amp_ckpt["discriminator"] if "discriminator" in amp_ckpt else amp_ckpt
            amp_discriminator.load_state_dict(amp_state)
            if amp_disc_trainer is not None and isinstance(amp_ckpt, dict):
                disc_opt = amp_ckpt.get("disc_optimizer") or amp_ckpt.get("optimizer")
                if disc_opt is not None:
                    amp_disc_trainer.optimizer.load_state_dict(disc_opt)
        amp_discriminator.eval()
        if args.amp_normalize_reward and amp_motion_buffer is not None:
            bootstrap_batch = min(
                int(args.amp_disc_batch_size),
                max(256, min(4096, int(amp_motion_buffer.num_transitions))),
            )
            real_obs_t, real_obs_t1 = amp_motion_buffer.sample_torch(bootstrap_batch, device=str(device))
            with torch.no_grad():
                real_scores = amp_discriminator(real_obs_t, real_obs_t1).detach().cpu().numpy()
            amp_d_real_anchor = float(np.mean(real_scores))
            amp_d_fake_anchor = float(np.mean(real_scores) - 1.0)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
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

    global_step = 0
    grad_updates = 0
    actor_updates = 0
    last_print_step = 0
    last_save_step = 0
    episodes_in_stage = 0

    completed_returns: List[float] = []
    completed_task_returns: List[float] = []
    completed_lengths: List[float] = []
    completed_successes: List[float] = []

    critic_loss_hist: List[float] = []
    actor_loss_hist: List[float] = []
    q_hist: List[float] = []
    amp_reward_hist: List[float] = []
    amp_weight_hist: List[float] = []
    amp_disc_loss_hist: List[float] = []
    amp_d_real_hist: List[float] = []
    amp_d_fake_hist: List[float] = []

    ep_return = 0.0
    ep_task_return = 0.0
    ep_len = 0
    start_time = time.time()

    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            env.close()
            logger.finish()
            raise FileNotFoundError(f"--resume-from checkpoint not found: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        ckpt_obs_dim = int(ckpt.get("obs_dim", obs_dim))
        ckpt_act_dim = int(ckpt.get("act_dim", act_dim))
        if ckpt_obs_dim != obs_dim or ckpt_act_dim != act_dim:
            env.close()
            logger.finish()
            raise ValueError(
                "Resume checkpoint dimensions do not match current run config: "
                f"ckpt(obs={ckpt_obs_dim}, act={ckpt_act_dim}) != run(obs={obs_dim}, act={act_dim})."
            )

        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        if "actor_target" in ckpt:
            actor_target.load_state_dict(ckpt["actor_target"])
        else:
            _hard_update(actor_target, actor)
        if "critic_target" in ckpt:
            critic_target.load_state_dict(ckpt["critic_target"])
        else:
            _hard_update(critic_target, critic)
        if "actor_optim" in ckpt:
            actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            critic_optim.load_state_dict(ckpt["critic_optim"])

        if args.amp_enable and amp_discriminator is not None and "amp_discriminator" in ckpt:
            amp_discriminator.load_state_dict(ckpt["amp_discriminator"])
            if (
                amp_disc_trainer is not None
                and "amp_disc_optimizer" in ckpt
                and ckpt["amp_disc_optimizer"] is not None
            ):
                amp_disc_trainer.optimizer.load_state_dict(ckpt["amp_disc_optimizer"])
        amp_d_real_anchor = ckpt.get("amp_d_real_anchor", amp_d_real_anchor)
        amp_d_fake_anchor = ckpt.get("amp_d_fake_anchor", amp_d_fake_anchor)

        global_step = int(ckpt.get("global_step", 0))
        grad_updates = int(ckpt.get("grad_updates", 0))
        actor_updates = int(ckpt.get("actor_updates", 0))
        last_print_step = int(ckpt.get("last_print_step", global_step))
        last_save_step = int(ckpt.get("last_save_step", global_step))
        episodes_in_stage = int(ckpt.get("episodes_in_stage", 0))
        current_stage = int(np.clip(int(ckpt.get("stage", current_stage)), 0, max_stage))

        env.close()
        env, obs_dict, _ = _build_env(args, stage=current_stage, seed=None)
        obs = np.asarray(obs_dict["h0"], dtype=np.float32)
        print(
            f"resumed from {args.resume_from}: global_step={global_step} "
            f"stage={current_stage} grad_updates={grad_updates} actor_updates={actor_updates}"
        )

    try:
        while global_step < args.total_steps:
            current_noise = _noise_std(args, global_step)
            if global_step < args.start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).cpu().numpy()
                if current_noise > 0.0:
                    action = action + np.random.normal(0.0, current_noise, size=action.shape)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

            amp_obs_t: Optional[np.ndarray] = None
            if args.amp_enable and amp_obs_builder is not None:
                amp_obs_t = amp_obs_builder.compute_obs(env.data, "h0")

            step_actions = {"h0": action, "h1": np.zeros_like(action, dtype=np.float32)}
            next_obs_dict, rewards, terminations, truncations, infos = env.step(step_actions)

            task_reward = float(rewards["h0"])
            combined_reward = task_reward
            amp_reward_scalar = 0.0
            amp_weight = 0.0

            if args.amp_enable:
                if (
                    amp_obs_builder is None
                    or amp_discriminator is None
                    or amp_policy_buffer is None
                    or amp_obs_t is None
                ):
                    raise RuntimeError("AMP is enabled but AMP components are not initialized.")

                amp_obs_t1 = amp_obs_builder.compute_obs(env.data, "h0")
                amp_policy_buffer.add(amp_obs_t, amp_obs_t1)

                obs_t_t = torch.as_tensor(amp_obs_t[None, :], dtype=torch.float32, device=device)
                obs_t1_t = torch.as_tensor(amp_obs_t1[None, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    d_score_t = amp_discriminator(obs_t_t, obs_t1_t)
                d_score_np = d_score_t.detach().cpu().numpy().astype(np.float32)

                if args.amp_normalize_reward:
                    amp_style = _normalized_amp_reward(
                        d_score=d_score_np,
                        d_real_anchor=amp_d_real_anchor,
                        d_fake_anchor=amp_d_fake_anchor,
                        reward_scale=float(args.amp_reward_scale),
                        clip_reward=bool(args.amp_clip_reward),
                    )
                    amp_d_fake_anchor = _update_running_anchor(
                        amp_d_fake_anchor,
                        float(np.mean(d_score_np)),
                        momentum=float(args.amp_reward_anchor_momentum),
                    )
                else:
                    with torch.no_grad():
                        amp_style_t = amp_discriminator.compute_reward(
                            obs_t_t,
                            obs_t1_t,
                            clip_reward=bool(args.amp_clip_reward),
                            reward_scale=float(args.amp_reward_scale),
                        )
                    amp_style = amp_style_t.detach().cpu().numpy().astype(np.float32)

                amp_reward_scalar = float(amp_style[0])
                amp_weight = _amp_weight_for_stage(args, current_stage)
                combined_reward = (1.0 - amp_weight) * task_reward + amp_weight * amp_reward_scalar
                amp_reward_hist.append(amp_reward_scalar)
                amp_weight_hist.append(amp_weight)
                if len(amp_reward_hist) > 20_000:
                    del amp_reward_hist[:-20_000]
                if len(amp_weight_hist) > 20_000:
                    del amp_weight_hist[:-20_000]

            done = bool(terminations["h0"] or truncations["h0"])
            next_obs = np.asarray(next_obs_dict["h0"], dtype=np.float32)
            replay.add(obs=obs, action=action, reward=combined_reward, next_obs=next_obs, done=done)

            ep_return += float(combined_reward)
            ep_task_return += float(task_reward)
            ep_len += 1
            global_step += 1

            if (
                args.amp_enable
                and amp_disc_trainer is not None
                and amp_motion_buffer is not None
                and amp_policy_buffer is not None
                and global_step >= args.amp_warmup_steps
                and amp_policy_buffer.size >= args.amp_disc_batch_size
            ):
                real_obs_t, real_obs_t1 = amp_motion_buffer.sample_torch(
                    args.amp_disc_batch_size, device=str(device)
                )
                fake_obs_t, fake_obs_t1 = amp_policy_buffer.sample_torch(
                    args.amp_disc_batch_size, device=str(device)
                )
                amp_disc_metrics = amp_disc_trainer.train_step(
                    real_obs_t, real_obs_t1, fake_obs_t, fake_obs_t1
                )
                amp_disc_loss_hist.append(float(amp_disc_metrics["disc_loss"]))
                amp_d_real_hist.append(float(amp_disc_metrics["d_real"]))
                amp_d_fake_hist.append(float(amp_disc_metrics["d_fake"]))
                if len(amp_disc_loss_hist) > 2_000:
                    del amp_disc_loss_hist[:-2_000]
                if len(amp_d_real_hist) > 2_000:
                    del amp_d_real_hist[:-2_000]
                if len(amp_d_fake_hist) > 2_000:
                    del amp_d_fake_hist[:-2_000]
                if args.amp_normalize_reward:
                    amp_d_real_anchor = _update_running_anchor(
                        amp_d_real_anchor,
                        float(amp_disc_metrics["d_real"]),
                        momentum=float(args.amp_reward_anchor_momentum),
                    )
                    amp_d_fake_anchor = _update_running_anchor(
                        amp_d_fake_anchor,
                        float(amp_disc_metrics["d_fake"]),
                        momentum=float(args.amp_reward_anchor_momentum),
                    )

            if (
                global_step >= args.update_after
                and replay.size >= args.batch_size
                and global_step % args.update_every == 0
            ):
                for _ in range(args.gradient_steps):
                    batch = replay.sample(args.batch_size, device=device)
                    with torch.no_grad():
                        next_actions = actor_target(batch.next_obs)
                        next_q = critic_target(batch.next_obs, next_actions)
                        q_target = batch.rewards + args.gamma * (1.0 - batch.dones) * next_q

                    q_pred = critic(batch.obs, batch.actions)
                    critic_loss = nn.functional.mse_loss(q_pred, q_target)
                    critic_optim.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                    critic_optim.step()

                    grad_updates += 1
                    critic_loss_hist.append(float(critic_loss.item()))
                    q_hist.append(float(q_pred.mean().item()))

                    if grad_updates % args.policy_delay == 0:
                        for p in critic.parameters():
                            p.requires_grad_(False)
                        pred_actions = actor(batch.obs)
                        actor_loss = -critic(batch.obs, pred_actions).mean()
                        if args.action_l2 > 0.0:
                            actor_loss = actor_loss + args.action_l2 * (pred_actions**2).mean()
                        actor_optim.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        actor_optim.step()
                        for p in critic.parameters():
                            p.requires_grad_(True)

                        actor_updates += 1
                        actor_loss_hist.append(float(actor_loss.item()))
                        _soft_update(actor_target, actor, args.tau)
                        _soft_update(critic_target, critic, args.tau)

            if done:
                termination_reason = str(infos["h0"].get("termination_reason", "unknown"))
                completed_returns.append(float(ep_return))
                completed_task_returns.append(float(ep_task_return))
                completed_lengths.append(float(ep_len))
                completed_successes.append(1.0 if termination_reason == "success" else 0.0)
                episodes_in_stage += 1

                ep_return = 0.0
                ep_task_return = 0.0
                ep_len = 0
                obs_dict, _ = env.reset(seed=None, options={"stage": int(current_stage)})
                obs = np.asarray(obs_dict["h0"], dtype=np.float32)
            else:
                obs = next_obs

            stage_up_msg = None
            if args.auto_curriculum and current_stage < max_stage:
                success_window = float(
                    np.mean(completed_successes[-args.curriculum_window_episodes :])
                ) if completed_successes else 0.0
                enough_episodes = len(completed_successes) >= args.curriculum_window_episodes
                enough_stage_episodes = episodes_in_stage >= args.curriculum_min_episodes_per_stage
                threshold = _resolve_curriculum_threshold(
                    current_stage=current_stage,
                    start_stage=int(args.stage),
                    thresholds=args.curriculum_thresholds,
                    default_threshold=args.curriculum_success_threshold,
                )
                if enough_episodes and enough_stage_episodes and success_window >= threshold:
                    prev_stage = current_stage
                    current_stage += 1
                    episodes_in_stage = 0
                    obs_dict, _ = env.reset(seed=None, options={"stage": int(current_stage)})
                    obs = np.asarray(obs_dict["h0"], dtype=np.float32)
                    stage_up_msg = (
                        f"curriculum: stage {prev_stage} -> {current_stage} "
                        f"(success_window={success_window:.2f} threshold={threshold:.2f})"
                    )

            if global_step - last_print_step >= args.print_every_steps or global_step >= args.total_steps:
                elapsed = max(1e-6, time.time() - start_time)
                sps = int(global_step / elapsed)
                mean_return = float(np.mean(completed_returns[-20:])) if completed_returns else 0.0
                mean_task_return = float(np.mean(completed_task_returns[-20:])) if completed_task_returns else 0.0
                mean_len = float(np.mean(completed_lengths[-20:])) if completed_lengths else 0.0
                mean_success_20 = float(np.mean(completed_successes[-20:])) if completed_successes else 0.0
                success_window = float(
                    np.mean(completed_successes[-args.curriculum_window_episodes :])
                ) if completed_successes else 0.0

                c_loss = float(np.mean(critic_loss_hist[-100:])) if critic_loss_hist else 0.0
                a_loss = float(np.mean(actor_loss_hist[-100:])) if actor_loss_hist else 0.0
                amp_reward_mean = float(np.mean(amp_reward_hist[-2000:])) if amp_reward_hist else 0.0
                amp_weight_mean = float(np.mean(amp_weight_hist[-2000:])) if amp_weight_hist else 0.0
                amp_disc_loss = float(np.mean(amp_disc_loss_hist[-100:])) if amp_disc_loss_hist else 0.0
                amp_d_real = float(np.mean(amp_d_real_hist[-100:])) if amp_d_real_hist else 0.0
                amp_d_fake = float(np.mean(amp_d_fake_hist[-100:])) if amp_d_fake_hist else 0.0

                msg = (
                    f"step={global_step}/{args.total_steps} sps={sps} stage={current_stage} "
                    f"ret20={mean_return:.2f} task20={mean_task_return:.2f} "
                    f"len20={mean_len:.1f} succ20={mean_success_20:.2f} succW={success_window:.2f} "
                    f"noise={current_noise:.3f} closs={c_loss:.4f} aloss={a_loss:.4f}"
                )
                if args.amp_enable:
                    msg += (
                        f" amp_w={amp_weight_mean:.2f} amp_r={amp_reward_mean:.3f} "
                        f"d_loss={amp_disc_loss:.4f} d_real={amp_d_real:.3f} d_fake={amp_d_fake:.3f}"
                    )
                print(msg)
                if stage_up_msg is not None:
                    print(stage_up_msg)
                last_print_step = global_step

                algo_metrics = {
                    "replay_size": replay.size,
                    "noise_std": current_noise,
                    "grad_updates": grad_updates,
                    "actor_updates": actor_updates,
                }
                if critic_loss_hist:
                    algo_metrics["critic_loss"] = c_loss
                if actor_loss_hist:
                    algo_metrics["actor_loss"] = a_loss
                if q_hist:
                    algo_metrics["q_mean"] = float(np.mean(q_hist[-100:]))
                if args.amp_enable:
                    algo_metrics["amp_weight_mean"] = amp_weight_mean
                    algo_metrics["amp_reward_mean"] = amp_reward_mean
                    algo_metrics["amp_disc_loss"] = amp_disc_loss
                    algo_metrics["amp_d_real"] = amp_d_real
                    algo_metrics["amp_d_fake"] = amp_d_fake
                    if args.amp_normalize_reward:
                        algo_metrics["amp_d_real_anchor"] = (
                            float(amp_d_real_anchor) if amp_d_real_anchor is not None else 0.0
                        )
                        algo_metrics["amp_d_fake_anchor"] = (
                            float(amp_d_fake_anchor) if amp_d_fake_anchor is not None else 0.0
                        )
                logger.log(
                    step=global_step,
                    common={
                        "sps": sps,
                        "ep_return_mean_20": mean_return,
                        "ep_task_return_mean_20": mean_task_return,
                        "ep_len_mean_20": mean_len,
                        "ep_success_mean_20": mean_success_20,
                        "ep_success_window": success_window,
                        "curriculum_stage": current_stage,
                    },
                    algo=algo_metrics,
                    task={"stage": float(current_stage)},
                )

            if global_step - last_save_step >= args.save_every_steps or global_step >= args.total_steps:
                payload = {
                    "algo": "ddpg_amp_single_agent_walk_to_target",
                    "args": vars(args),
                    "global_step": global_step,
                    "stage": current_stage,
                    "grad_updates": grad_updates,
                    "actor_updates": actor_updates,
                    "episodes_in_stage": episodes_in_stage,
                    "last_print_step": last_print_step,
                    "last_save_step": last_save_step,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_target": actor_target.state_dict(),
                    "critic_target": critic_target.state_dict(),
                    "actor_optim": actor_optim.state_dict(),
                    "critic_optim": critic_optim.state_dict(),
                    "amp_d_real_anchor": amp_d_real_anchor,
                    "amp_d_fake_anchor": amp_d_fake_anchor,
                }
                if args.amp_enable and amp_discriminator is not None:
                    payload["amp_discriminator"] = amp_discriminator.state_dict()
                    payload["amp_obs_dim"] = int(amp_obs_dim)
                    payload["amp_disc_optimizer"] = (
                        amp_disc_trainer.optimizer.state_dict() if amp_disc_trainer is not None else None
                    )
                ckpt_path = os.path.join(args.save_dir, f"ddpg_amp_walk_step_{global_step:07d}.pt")
                latest_path = os.path.join(args.save_dir, "latest.pt")
                torch.save(payload, ckpt_path)
                torch.save(payload, latest_path)
                last_save_step = global_step

    finally:
        env.close()
        logger.finish()

    print("Training complete.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
