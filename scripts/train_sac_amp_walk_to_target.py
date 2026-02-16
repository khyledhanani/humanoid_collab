#!/usr/bin/env python3
"""Train single-agent SAC on walk_to_target with AMP style reward blending."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError as exc:
    raise ImportError(
        "SAC training requires PyTorch. Install training dependencies with: "
        "pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.amp.amp_obs import AMPObsBuilder, quat_to_mat
from humanoid_collab.amp.discriminator import AMPDiscriminator, AMPDiscriminatorTrainer
from humanoid_collab.amp.motion_buffer import MotionReplayBuffer, PolicyTransitionBuffer
from humanoid_collab.amp.motion_data import load_motion_dataset
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger
from humanoid_collab.vector_env import (
    SharedMemHumanoidCollabVecEnv,
    SubprocHumanoidCollabVecEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train single-agent SAC+AMP on walk_to_target."
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
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env-backend",
        type=str,
        default="shared_memory",
        choices=["shared_memory", "subproc"],
        help="Vector env backend used when --num-envs > 1.",
    )
    parser.add_argument("--vec-backend", dest="vec_env_backend", choices=["shared_memory", "subproc"], help=argparse.SUPPRESS)
    parser.add_argument("--start-method", type=str, default=None)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=2_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--autotune-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically tune entropy temperature alpha.",
    )
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="Target policy entropy. Defaults to -action_dim.",
    )
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

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
    parser.add_argument("--amp-weight-stage0", type=float, default=0.50)
    parser.add_argument("--amp-weight-stage1", type=float, default=0.40)
    parser.add_argument("--amp-weight-stage2", type=float, default=0.35)
    parser.add_argument("--amp-weight-stage3", type=float, default=0.30)
    parser.add_argument(
        "--amp-anneal-steps",
        type=int,
        default=0,
        help="If > 0, linearly anneal AMP reward weight over this many env steps.",
    )
    parser.add_argument(
        "--amp-anneal-begin-step",
        type=int,
        default=0,
        help="Env step at which AMP annealing starts (default: 0).",
    )
    parser.add_argument(
        "--amp-anneal-start-weight",
        type=float,
        default=None,
        help="Starting AMP weight for anneal (default: stage weight or override).",
    )
    parser.add_argument(
        "--amp-anneal-end-weight",
        type=float,
        default=None,
        help="Final AMP weight for anneal (default: 0.0).",
    )

    parser.add_argument("--auto-curriculum", action="store_true")
    parser.add_argument("--curriculum-window-episodes", type=int, default=80)
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.70)
    parser.add_argument("--curriculum-thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--curriculum-min-episodes-per-stage", type=int, default=120)

    parser.add_argument("--log-dir", type=str, default="runs/sac_amp_walk_to_target")
    parser.add_argument("--save-dir", type=str, default="checkpoints/sac_amp_walk_to_target")
    parser.add_argument("--save-every-steps", type=int, default=50_000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--print-every-steps", type=int, default=2_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="sac-amp-walk")
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


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        # Tanh-squash correction for change of variables.
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mu)
        return action, log_prob, mean_action


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


def _amp_weight(args: argparse.Namespace, stage: int, global_step: int) -> float:
    """Compute AMP blend weight, optionally linearly annealed over env steps."""
    base = _amp_weight_for_stage(args, stage)
    anneal_steps = int(getattr(args, "amp_anneal_steps", 0) or 0)
    if anneal_steps <= 0:
        return float(np.clip(base, 0.0, 1.0))

    begin = int(getattr(args, "amp_anneal_begin_step", 0) or 0)
    start_w = getattr(args, "amp_anneal_start_weight", None)
    end_w = getattr(args, "amp_anneal_end_weight", None)
    start = float(base if start_w is None else start_w)
    end = float(0.0 if end_w is None else end_w)

    t = max(0, int(global_step) - begin)
    frac = float(np.clip(t / max(1, anneal_steps), 0.0, 1.0))
    w = start + frac * (end - start)
    return float(np.clip(w, 0.0, 1.0))


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


def _make_env_kwargs(args: argparse.Namespace, stage: int) -> Dict[str, object]:
    return dict(
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


def _format_reset_obs(obs: Dict[str, np.ndarray], num_envs: int) -> Dict[str, np.ndarray]:
    if num_envs == 1:
        return {
            agent: np.asarray(obs[agent], dtype=np.float32)[None, ...]
            for agent in ("h0", "h1")
        }
    return {
        agent: np.asarray(obs[agent], dtype=np.float32)
        for agent in ("h0", "h1")
    }


def _normalize_infos_payload(infos_payload: Any, num_envs: int) -> List[Dict[str, Dict[str, Any]]]:
    if num_envs == 1 and isinstance(infos_payload, dict):
        return [infos_payload]
    if not isinstance(infos_payload, list):
        raise TypeError(f"Expected list infos payload for num_envs={num_envs}, got {type(infos_payload)}")
    return infos_payload


def _extract_done_and_reasons_single(
    terminations: Dict[str, bool],
    truncations: Dict[str, bool],
    infos: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    done = bool(terminations["h0"] or truncations["h0"])
    reason = str(infos.get("h0", {}).get("termination_reason", "running"))
    if done and reason == "running":
        reason = "termination" if bool(terminations["h0"]) else "truncation"
    return np.asarray([done], dtype=np.bool_), [reason]


def _extract_done_and_reasons_vec(
    terminations: Dict[str, np.ndarray],
    truncations: Dict[str, np.ndarray],
    infos_list: List[Dict[str, Dict[str, Any]]],
) -> Tuple[np.ndarray, List[str]]:
    term_h0 = np.asarray(terminations["h0"], dtype=np.bool_)
    trunc_h0 = np.asarray(truncations["h0"], dtype=np.bool_)
    done_mask = np.logical_or(term_h0, trunc_h0)
    reasons: List[str] = []
    for env_idx, done in enumerate(done_mask):
        info_env = infos_list[env_idx] if env_idx < len(infos_list) else {}
        reason = str(info_env.get("h0", {}).get("termination_reason", "running"))
        if bool(done) and reason == "running":
            reason = "termination" if bool(term_h0[env_idx]) else "truncation"
        reasons.append(reason)
    return done_mask, reasons


def _maybe_replace_final_obs_from_infos(
    next_obs_batch: np.ndarray,
    done_mask: np.ndarray,
    infos: List[Dict[str, Dict[str, Any]]],
) -> np.ndarray:
    out = np.asarray(next_obs_batch, dtype=np.float32).copy()
    for env_idx, done in enumerate(done_mask):
        if not bool(done):
            continue
        info_env = infos[env_idx] if env_idx < len(infos) else {}
        final_obs = info_env.get("h0", {}).get("final_observation")
        if final_obs is not None:
            out[env_idx] = np.asarray(final_obs, dtype=np.float32)
    return out


def _extract_root_height_from_infos(
    infos_list: List[Dict[str, Dict[str, Any]]],
    num_envs: int,
    done_mask: Optional[np.ndarray],
    use_reset_info_for_done: bool,
) -> np.ndarray:
    if done_mask is None:
        done_mask = np.zeros((num_envs,), dtype=np.bool_)
    values: List[float] = []
    for env_idx in range(num_envs):
        info_env = infos_list[env_idx]
        info_src = info_env.get("h0", {})
        if use_reset_info_for_done and bool(done_mask[env_idx]):
            reset_info = info_src.get("reset_info")
            if isinstance(reset_info, dict):
                info_src = reset_info
        if "root_height" not in info_src:
            raise KeyError("Expected 'root_height' in infos payload for AMP training.")
        values.append(float(info_src["root_height"]))
    return np.asarray(values, dtype=np.float32)


def _build_amp_obs_from_proprio(
    proprio_batch: np.ndarray,
    root_height_batch: np.ndarray,
    joint_qpos_dim: int,
    joint_qvel_dim: int,
    include_root_height: bool,
) -> np.ndarray:
    proprio = np.asarray(proprio_batch, dtype=np.float32)
    if proprio.ndim != 2:
        raise ValueError(f"Expected 2D proprio batch, got {proprio.shape}")
    base_dim = joint_qpos_dim + joint_qvel_dim + 4 + 3
    if proprio.shape[1] < base_dim:
        raise ValueError(
            f"Proprio dim is {proprio.shape[1]}, expected at least {base_dim} "
            f"(joint_qpos={joint_qpos_dim}, joint_qvel={joint_qvel_dim})."
        )

    n = proprio.shape[0]
    offset = 0
    joint_qpos = proprio[:, offset : offset + joint_qpos_dim]
    offset += joint_qpos_dim
    joint_qvel = proprio[:, offset : offset + joint_qvel_dim]
    offset += joint_qvel_dim
    root_quat = proprio[:, offset : offset + 4]
    offset += 4
    root_angvel = proprio[:, offset : offset + 3]

    fwd = np.zeros((n, 3), dtype=np.float32)
    up = np.zeros((n, 3), dtype=np.float32)
    for idx in range(n):
        xmat = quat_to_mat(root_quat[idx])
        fwd[idx] = xmat[:, 0].astype(np.float32)
        up[idx] = xmat[:, 2].astype(np.float32)

    parts: List[np.ndarray] = [joint_qpos, joint_qvel]
    if include_root_height:
        parts.append(root_height_batch.reshape(-1, 1))
    parts.extend([fwd, up, root_angvel])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _compute_amp_obs_batch(
    *,
    raw_obs_batch: np.ndarray,
    infos_payload: Any,
    num_envs: int,
    done_mask: Optional[np.ndarray],
    use_reset_info_for_done: bool,
    joint_qpos_dim: int,
    joint_qvel_dim: int,
    include_root_height: bool,
) -> np.ndarray:
    infos_list = _normalize_infos_payload(infos_payload, num_envs)
    root_height = _extract_root_height_from_infos(
        infos_list=infos_list,
        num_envs=num_envs,
        done_mask=done_mask,
        use_reset_info_for_done=use_reset_info_for_done,
    )
    return _build_amp_obs_from_proprio(
        proprio_batch=np.asarray(raw_obs_batch, dtype=np.float32),
        root_height_batch=root_height,
        joint_qpos_dim=joint_qpos_dim,
        joint_qvel_dim=joint_qvel_dim,
        include_root_height=include_root_height,
    )


def _build_env(args: argparse.Namespace, stage: int, seed: Optional[int]):
    env_kwargs = _make_env_kwargs(args, stage)
    if args.num_envs == 1:
        env = HumanoidCollabEnv(**env_kwargs)
        obs, infos = env.reset(seed=seed, options={"stage": int(stage)})
    else:
        if args.vec_env_backend == "shared_memory":
            env = SharedMemHumanoidCollabVecEnv(
                num_envs=args.num_envs,
                env_kwargs=env_kwargs,
                auto_reset=True,
                start_method=args.start_method,
            )
        else:
            env = SubprocHumanoidCollabVecEnv(
                num_envs=args.num_envs,
                env_kwargs=env_kwargs,
                auto_reset=True,
                start_method=args.start_method,
            )
        obs, infos = env.reset(seed=seed, options={"stage": int(stage)})
    return env, _format_reset_obs(obs, args.num_envs), infos


def train(args: argparse.Namespace) -> None:
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")
    if args.buffer_size <= args.batch_size:
        raise ValueError("--buffer-size should be larger than --batch-size")
    if args.curriculum_window_episodes <= 0:
        raise ValueError("--curriculum-window-episodes must be > 0")
    if args.curriculum_min_episodes_per_stage <= 0:
        raise ValueError("--curriculum-min-episodes-per-stage must be > 0")
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
    env, obs_dict, reset_infos = _build_env(args, stage=current_stage, seed=args.seed)
    obs = np.asarray(obs_dict["h0"], dtype=np.float32)
    obs_dim = int(obs.shape[-1])
    num_envs = int(obs.shape[0])
    act_dim = int(env.action_space("h0").shape[0])

    actor = GaussianActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_size,
        log_std_min=float(args.log_std_min),
        log_std_max=float(args.log_std_max),
    ).to(device)
    critic1 = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    critic2 = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    critic1_target = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    critic2_target = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size).to(device)
    _hard_update(critic1_target, critic1)
    _hard_update(critic2_target, critic2)

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = optim.Adam(critic2.parameters(), lr=args.critic_lr)

    log_alpha = torch.tensor(
        np.log(max(1e-8, float(args.alpha))),
        dtype=torch.float32,
        device=device,
        requires_grad=bool(args.autotune_alpha),
    )
    alpha_optim: Optional[optim.Optimizer] = None
    if args.autotune_alpha:
        alpha_optim = optim.Adam([log_alpha], lr=float(args.alpha_lr))
    target_entropy = float(args.target_entropy) if args.target_entropy is not None else -float(act_dim)
    replay = ReplayBuffer(capacity=args.buffer_size, obs_dim=obs_dim, act_dim=act_dim)

    amp_motion_buffer: Optional[MotionReplayBuffer] = None
    amp_policy_buffer: Optional[PolicyTransitionBuffer] = None
    amp_discriminator: Optional[AMPDiscriminator] = None
    amp_disc_trainer: Optional[AMPDiscriminatorTrainer] = None
    amp_obs_builder: Optional[AMPObsBuilder] = None
    amp_joint_qpos_dim = 0
    amp_joint_qvel_dim = 0
    amp_obs_dim = 0
    amp_obs_current: Optional[np.ndarray] = None
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

        sample_clip = motion_dataset.clips[0]
        nq_agent = int(sample_clip.qpos.shape[1])
        nv_agent = int(sample_clip.qvel.shape[1])
        amp_joint_qpos_dim = nq_agent - 7
        amp_joint_qvel_dim = nv_agent - 6

        if isinstance(env, HumanoidCollabEnv):
            amp_obs_builder = AMPObsBuilder(
                id_cache=env.id_cache,
                include_root_height=bool(args.amp_include_root_height),
                include_root_orientation=True,
                include_joint_positions=True,
                include_joint_velocities=True,
            )
        else:
            tmp_env = HumanoidCollabEnv(**_make_env_kwargs(args, current_stage))
            try:
                amp_obs_builder = AMPObsBuilder(
                    id_cache=tmp_env.id_cache,
                    include_root_height=bool(args.amp_include_root_height),
                    include_root_orientation=True,
                    include_joint_positions=True,
                    include_joint_velocities=True,
                )
            finally:
                tmp_env.close()

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
        amp_obs_current = _compute_amp_obs_batch(
            raw_obs_batch=obs,
            infos_payload=reset_infos,
            num_envs=num_envs,
            done_mask=None,
            use_reset_info_for_done=False,
            joint_qpos_dim=amp_joint_qpos_dim,
            joint_qvel_dim=amp_joint_qvel_dim,
            include_root_height=bool(args.amp_include_root_height),
        )

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
    collector_steps = 0
    grad_updates = 0
    actor_updates = 0
    last_print_step = 0
    last_save_step = 0
    episodes_in_stage = 0

    completed_returns: List[float] = []
    completed_task_returns: List[float] = []
    completed_lengths: List[float] = []
    completed_successes: List[float] = []

    critic1_loss_hist: List[float] = []
    critic2_loss_hist: List[float] = []
    actor_loss_hist: List[float] = []
    alpha_loss_hist: List[float] = []
    q1_hist: List[float] = []
    q2_hist: List[float] = []
    alpha_hist: List[float] = []
    amp_reward_hist: List[float] = []
    amp_weight_hist: List[float] = []
    amp_weighted_reward_hist: List[float] = []
    task_weighted_reward_hist: List[float] = []
    amp_disc_loss_hist: List[float] = []
    amp_d_real_hist: List[float] = []
    amp_d_fake_hist: List[float] = []
    amp_weight_eff_hist: List[float] = []

    ep_return = np.zeros((num_envs,), dtype=np.float32)
    ep_task_return = np.zeros((num_envs,), dtype=np.float32)
    ep_len = np.zeros((num_envs,), dtype=np.int32)
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
        if "critic1" not in ckpt or "critic2" not in ckpt:
            env.close()
            logger.finish()
            raise ValueError(
                "Checkpoint is not SAC-compatible (missing critic1/critic2). "
                "Use a checkpoint produced by train_sac_amp_walk_to_target.py."
            )

        actor.load_state_dict(ckpt["actor"])
        critic1.load_state_dict(ckpt["critic1"])
        critic2.load_state_dict(ckpt["critic2"])
        if "critic1_target" in ckpt:
            critic1_target.load_state_dict(ckpt["critic1_target"])
        else:
            _hard_update(critic1_target, critic1)
        if "critic2_target" in ckpt:
            critic2_target.load_state_dict(ckpt["critic2_target"])
        else:
            _hard_update(critic2_target, critic2)
        if "actor_optim" in ckpt:
            actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic1_optim" in ckpt:
            critic1_optim.load_state_dict(ckpt["critic1_optim"])
        if "critic2_optim" in ckpt:
            critic2_optim.load_state_dict(ckpt["critic2_optim"])
        if "log_alpha" in ckpt:
            log_alpha.data.copy_(torch.as_tensor(ckpt["log_alpha"], dtype=torch.float32, device=device))
        if alpha_optim is not None and "alpha_optim" in ckpt and ckpt["alpha_optim"] is not None:
            alpha_optim.load_state_dict(ckpt["alpha_optim"])

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
        collector_steps = int(ckpt.get("collector_steps", global_step // max(1, num_envs)))
        grad_updates = int(ckpt.get("grad_updates", 0))
        actor_updates = int(ckpt.get("actor_updates", 0))
        last_print_step = int(ckpt.get("last_print_step", global_step))
        last_save_step = int(ckpt.get("last_save_step", global_step))
        episodes_in_stage = int(ckpt.get("episodes_in_stage", 0))
        current_stage = int(np.clip(int(ckpt.get("stage", current_stage)), 0, max_stage))

        env.close()
        env, obs_dict, reset_infos = _build_env(args, stage=current_stage, seed=None)
        obs = np.asarray(obs_dict["h0"], dtype=np.float32)
        num_envs = int(obs.shape[0])
        if args.amp_enable:
            amp_obs_current = _compute_amp_obs_batch(
                raw_obs_batch=obs,
                infos_payload=reset_infos,
                num_envs=num_envs,
                done_mask=None,
                use_reset_info_for_done=False,
                joint_qpos_dim=amp_joint_qpos_dim,
                joint_qvel_dim=amp_joint_qvel_dim,
                include_root_height=bool(args.amp_include_root_height),
            )
        print(
            f"resumed from {args.resume_from}: global_step={global_step} "
            f"stage={current_stage} grad_updates={grad_updates} actor_updates={actor_updates} "
            f"alpha={float(log_alpha.exp().item()):.4f}"
        )

    try:
        while global_step < args.total_steps:
            if global_step < args.start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_t, _, _ = actor.sample(obs_t)
                action = action_t.cpu().numpy().astype(np.float32)

            if num_envs == 1:
                step_actions = {"h0": action[0], "h1": np.zeros_like(action[0], dtype=np.float32)}
                next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(step_actions)
                done_mask, reasons = _extract_done_and_reasons_single(terminations, truncations, infos)
                raw_next_obs_for_buffer = np.asarray(next_obs_dict["h0"], dtype=np.float32)[None, ...]
                raw_next_obs_for_rollout = raw_next_obs_for_buffer
                infos_list = [infos]
                rollout_infos_payload: Any = infos_list
                task_reward_batch = np.asarray([float(rewards_dict["h0"])], dtype=np.float32)

                if bool(done_mask[0]):
                    reset_obs, reset_infos = env.reset(seed=None, options={"stage": int(current_stage)})
                    raw_next_obs_for_rollout = np.asarray(reset_obs["h0"], dtype=np.float32)[None, ...]
                    rollout_infos_payload = reset_infos
            else:
                step_actions = {"h0": action, "h1": np.zeros_like(action, dtype=np.float32)}
                next_obs_vec, rewards_vec, terminations, truncations, infos_list = env.step(step_actions)
                done_mask, reasons = _extract_done_and_reasons_vec(terminations, truncations, infos_list)
                raw_next_obs_for_rollout = np.asarray(next_obs_vec["h0"], dtype=np.float32)
                raw_next_obs_for_buffer = _maybe_replace_final_obs_from_infos(
                    next_obs_batch=raw_next_obs_for_rollout,
                    done_mask=done_mask,
                    infos=infos_list,
                )
                rollout_infos_payload = infos_list
                task_reward_batch = np.asarray(rewards_vec["h0"], dtype=np.float32)

            combined_reward_batch = task_reward_batch.copy()
            amp_weight = 0.0

            if args.amp_enable:
                if (
                    amp_obs_current is None
                    or amp_discriminator is None
                    or amp_policy_buffer is None
                ):
                    raise RuntimeError("AMP is enabled but AMP components are not initialized.")

                amp_obs_t = amp_obs_current
                amp_obs_t1_for_buffer = _compute_amp_obs_batch(
                    raw_obs_batch=raw_next_obs_for_buffer,
                    infos_payload=infos_list,
                    num_envs=num_envs,
                    done_mask=done_mask,
                    use_reset_info_for_done=False,
                    joint_qpos_dim=amp_joint_qpos_dim,
                    joint_qvel_dim=amp_joint_qvel_dim,
                    include_root_height=bool(args.amp_include_root_height),
                )
                amp_policy_buffer.add_batch(amp_obs_t, amp_obs_t1_for_buffer)

                obs_t_t = torch.as_tensor(amp_obs_t, dtype=torch.float32, device=device)
                obs_t1_t = torch.as_tensor(amp_obs_t1_for_buffer, dtype=torch.float32, device=device)
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

                amp_weight = _amp_weight(args, current_stage, global_step)

                # Stability-gate AMP weight so the discriminator cannot reward unstable falling
                # behaviors throughout an episode (not just the terminal fall transition).
                tilt_vals: List[float] = []
                height_vals: List[float] = []
                for info_env in infos_list:
                    h0_info = info_env.get("h0", {}) if isinstance(info_env, dict) else {}
                    tilt_vals.append(float(h0_info.get("walk_tilt", 0.0)))
                    height_vals.append(float(h0_info.get("root_height", 0.0)))
                tilt = np.asarray(tilt_vals, dtype=np.float32)
                root_h = np.asarray(height_vals, dtype=np.float32)

                # Use the same upright threshold as the task's success condition.
                tilt_gate = np.clip((0.55 - tilt) / 0.55, 0.0, 1.0)
                height_gate = np.clip((root_h - 0.5) / 0.35, 0.0, 1.0)
                stability_gate = (tilt_gate * height_gate).astype(np.float32)

                amp_weight_eff = float(amp_weight) * stability_gate
                combined_reward_batch = (
                    (1.0 - amp_weight_eff) * task_reward_batch + amp_weight_eff * amp_style
                ).astype(np.float32)
                amp_reward_hist.extend(amp_style.tolist())
                amp_weight_hist.extend([float(amp_weight)] * num_envs)
                amp_weight_eff_hist.extend(amp_weight_eff.tolist())
                amp_weighted_reward_hist.extend((amp_weight * amp_style).tolist())
                task_weighted_reward_hist.extend(((1.0 - amp_weight) * task_reward_batch).tolist())
                if len(amp_reward_hist) > 20_000:
                    del amp_reward_hist[:-20_000]
                if len(amp_weight_hist) > 20_000:
                    del amp_weight_hist[:-20_000]
                if len(amp_weight_eff_hist) > 20_000:
                    del amp_weight_eff_hist[:-20_000]
                if len(amp_weighted_reward_hist) > 20_000:
                    del amp_weighted_reward_hist[:-20_000]
                if len(task_weighted_reward_hist) > 20_000:
                    del task_weighted_reward_hist[:-20_000]

                amp_obs_current = _compute_amp_obs_batch(
                    raw_obs_batch=raw_next_obs_for_rollout,
                    infos_payload=rollout_infos_payload,
                    num_envs=num_envs,
                    done_mask=done_mask,
                    use_reset_info_for_done=(num_envs > 1),
                    joint_qpos_dim=amp_joint_qpos_dim,
                    joint_qvel_dim=amp_joint_qvel_dim,
                    include_root_height=bool(args.amp_include_root_height),
                )

            for env_idx in range(num_envs):
                replay.add(
                    obs=obs[env_idx],
                    action=action[env_idx],
                    reward=float(combined_reward_batch[env_idx]),
                    next_obs=raw_next_obs_for_buffer[env_idx],
                    done=bool(done_mask[env_idx]),
                )

            ep_return += combined_reward_batch
            ep_task_return += task_reward_batch
            ep_len += 1
            global_step += num_envs
            collector_steps += 1

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
                and collector_steps % args.update_every == 0
            ):
                for _ in range(args.gradient_steps):
                    batch = replay.sample(args.batch_size, device=device)
                    alpha_t = log_alpha.exp()

                    with torch.no_grad():
                        next_actions, next_log_prob, _ = actor.sample(batch.next_obs)
                        next_q1 = critic1_target(batch.next_obs, next_actions)
                        next_q2 = critic2_target(batch.next_obs, next_actions)
                        next_q = torch.min(next_q1, next_q2) - alpha_t * next_log_prob
                        q_target = batch.rewards + args.gamma * (1.0 - batch.dones) * next_q

                    q1_pred = critic1(batch.obs, batch.actions)
                    q2_pred = critic2(batch.obs, batch.actions)
                    critic1_loss = F.mse_loss(q1_pred, q_target)
                    critic2_loss = F.mse_loss(q2_pred, q_target)

                    critic1_optim.zero_grad(set_to_none=True)
                    critic1_loss.backward()
                    nn.utils.clip_grad_norm_(critic1.parameters(), args.max_grad_norm)
                    critic1_optim.step()

                    critic2_optim.zero_grad(set_to_none=True)
                    critic2_loss.backward()
                    nn.utils.clip_grad_norm_(critic2.parameters(), args.max_grad_norm)
                    critic2_optim.step()

                    grad_updates += 1
                    critic1_loss_hist.append(float(critic1_loss.item()))
                    critic2_loss_hist.append(float(critic2_loss.item()))
                    q1_hist.append(float(q1_pred.mean().item()))
                    q2_hist.append(float(q2_pred.mean().item()))
                    if len(critic1_loss_hist) > 20_000:
                        del critic1_loss_hist[:-20_000]
                    if len(critic2_loss_hist) > 20_000:
                        del critic2_loss_hist[:-20_000]
                    if len(q1_hist) > 20_000:
                        del q1_hist[:-20_000]
                    if len(q2_hist) > 20_000:
                        del q2_hist[:-20_000]

                    pi_actions, log_prob, _ = actor.sample(batch.obs)
                    q1_pi = critic1(batch.obs, pi_actions)
                    q2_pi = critic2(batch.obs, pi_actions)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha_t.detach() * log_prob - q_pi).mean()

                    actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    actor_optim.step()

                    actor_updates += 1
                    actor_loss_hist.append(float(actor_loss.item()))
                    if len(actor_loss_hist) > 20_000:
                        del actor_loss_hist[:-20_000]

                    if args.autotune_alpha and alpha_optim is not None:
                        alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                        alpha_optim.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_optim.step()
                        alpha_loss_hist.append(float(alpha_loss.item()))
                        if len(alpha_loss_hist) > 20_000:
                            del alpha_loss_hist[:-20_000]

                    alpha_hist.append(float(log_alpha.exp().item()))
                    if len(alpha_hist) > 20_000:
                        del alpha_hist[:-20_000]

                    _soft_update(critic1_target, critic1, args.tau)
                    _soft_update(critic2_target, critic2, args.tau)

            for env_idx, done in enumerate(done_mask):
                if not bool(done):
                    continue
                completed_returns.append(float(ep_return[env_idx]))
                completed_task_returns.append(float(ep_task_return[env_idx]))
                completed_lengths.append(float(ep_len[env_idx]))
                completed_successes.append(1.0 if reasons[env_idx] == "success" else 0.0)
                episodes_in_stage += 1

                ep_return[env_idx] = 0.0
                ep_task_return[env_idx] = 0.0
                ep_len[env_idx] = 0

            obs = raw_next_obs_for_rollout

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
                    reset_obs, reset_infos = env.reset(seed=None, options={"stage": int(current_stage)})
                    obs = np.asarray(_format_reset_obs(reset_obs, num_envs)["h0"], dtype=np.float32)
                    if args.amp_enable:
                        amp_obs_current = _compute_amp_obs_batch(
                            raw_obs_batch=obs,
                            infos_payload=reset_infos,
                            num_envs=num_envs,
                            done_mask=None,
                            use_reset_info_for_done=False,
                            joint_qpos_dim=amp_joint_qpos_dim,
                            joint_qvel_dim=amp_joint_qvel_dim,
                            include_root_height=bool(args.amp_include_root_height),
                        )
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

                c1_loss = float(np.mean(critic1_loss_hist[-100:])) if critic1_loss_hist else 0.0
                c2_loss = float(np.mean(critic2_loss_hist[-100:])) if critic2_loss_hist else 0.0
                a_loss = float(np.mean(actor_loss_hist[-100:])) if actor_loss_hist else 0.0
                alpha_loss_mean = float(np.mean(alpha_loss_hist[-100:])) if alpha_loss_hist else 0.0
                alpha_value = float(log_alpha.exp().item())
                amp_reward_mean = float(np.mean(amp_reward_hist[-2000:])) if amp_reward_hist else 0.0
                amp_weight_mean = float(np.mean(amp_weight_hist[-2000:])) if amp_weight_hist else 0.0
                amp_weight_eff_mean = (
                    float(np.mean(amp_weight_eff_hist[-2000:])) if amp_weight_eff_hist else 0.0
                )
                amp_weighted_mean = (
                    float(np.mean(amp_weighted_reward_hist[-2000:])) if amp_weighted_reward_hist else 0.0
                )
                task_weighted_mean = (
                    float(np.mean(task_weighted_reward_hist[-2000:])) if task_weighted_reward_hist else 0.0
                )
                amp_disc_loss = float(np.mean(amp_disc_loss_hist[-100:])) if amp_disc_loss_hist else 0.0
                amp_d_real = float(np.mean(amp_d_real_hist[-100:])) if amp_d_real_hist else 0.0
                amp_d_fake = float(np.mean(amp_d_fake_hist[-100:])) if amp_d_fake_hist else 0.0

                msg = (
                    f"step={global_step}/{args.total_steps} sps={sps} stage={current_stage} "
                    f"ret20={mean_return:.2f} task20={mean_task_return:.2f} "
                    f"len20={mean_len:.1f} succ20={mean_success_20:.2f} succW={success_window:.2f} "
                    f"alpha={alpha_value:.4f} c1={c1_loss:.4f} c2={c2_loss:.4f} aloss={a_loss:.4f}"
                )
                if args.amp_enable:
                    msg += (
                        f" amp_w={amp_weight_mean:.2f} amp_w_eff={amp_weight_eff_mean:.2f} amp_r={amp_reward_mean:.3f} "
                        f"amp_w*r={amp_weighted_mean:.3f} task_w*r={task_weighted_mean:.3f} "
                        f"d_loss={amp_disc_loss:.4f} d_real={amp_d_real:.3f} d_fake={amp_d_fake:.3f}"
                    )
                print(msg)
                if stage_up_msg is not None:
                    print(stage_up_msg)
                last_print_step = global_step

                algo_metrics = {
                    "replay_size": replay.size,
                    "collector_steps": collector_steps,
                    "grad_updates": grad_updates,
                    "actor_updates": actor_updates,
                    "alpha": alpha_value,
                }
                if critic1_loss_hist:
                    algo_metrics["critic1_loss"] = c1_loss
                if critic2_loss_hist:
                    algo_metrics["critic2_loss"] = c2_loss
                if actor_loss_hist:
                    algo_metrics["actor_loss"] = a_loss
                if alpha_loss_hist:
                    algo_metrics["alpha_loss"] = alpha_loss_mean
                if q1_hist:
                    algo_metrics["q1_mean"] = float(np.mean(q1_hist[-100:]))
                if q2_hist:
                    algo_metrics["q2_mean"] = float(np.mean(q2_hist[-100:]))
                if args.amp_enable:
                    algo_metrics["amp_weight_mean"] = amp_weight_mean
                    algo_metrics["amp_weight_eff_mean"] = amp_weight_eff_mean
                    algo_metrics["amp_reward_mean"] = amp_reward_mean
                    algo_metrics["amp_weighted_reward_mean"] = amp_weighted_mean
                    algo_metrics["task_weighted_reward_mean"] = task_weighted_mean
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
                    "algo": "sac_amp_single_agent_walk_to_target",
                    "args": vars(args),
                    "global_step": global_step,
                    "collector_steps": collector_steps,
                    "stage": current_stage,
                    "grad_updates": grad_updates,
                    "actor_updates": actor_updates,
                    "episodes_in_stage": episodes_in_stage,
                    "last_print_step": last_print_step,
                    "last_save_step": last_save_step,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "actor": actor.state_dict(),
                    "critic1": critic1.state_dict(),
                    "critic2": critic2.state_dict(),
                    "critic1_target": critic1_target.state_dict(),
                    "critic2_target": critic2_target.state_dict(),
                    "actor_optim": actor_optim.state_dict(),
                    "critic1_optim": critic1_optim.state_dict(),
                    "critic2_optim": critic2_optim.state_dict(),
                    "log_alpha": float(log_alpha.detach().cpu().item()),
                    "alpha_optim": alpha_optim.state_dict() if alpha_optim is not None else None,
                    "amp_d_real_anchor": amp_d_real_anchor,
                    "amp_d_fake_anchor": amp_d_fake_anchor,
                }
                if args.amp_enable and amp_discriminator is not None:
                    payload["amp_discriminator"] = amp_discriminator.state_dict()
                    payload["amp_obs_dim"] = int(amp_obs_dim)
                    payload["amp_disc_optimizer"] = (
                        amp_disc_trainer.optimizer.state_dict() if amp_disc_trainer is not None else None
                    )
                ckpt_path = os.path.join(args.save_dir, f"sac_amp_walk_step_{global_step:07d}.pt")
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
