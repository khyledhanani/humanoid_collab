"""Train MADDPG policies for HumanoidCollabEnv.

Implements centralized training with decentralized execution (CTDE):
- Actor per agent consumes only that agent's local observation.
- Critic per agent consumes joint observations and joint actions.
"""

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
    import torch.optim as optim
except ImportError as exc:
    raise ImportError(
        "MADDPG training requires PyTorch. Install training dependencies with: "
        "pip install -e '.[train]'"
    ) from exc

from humanoid_collab import (
    HumanoidCollabEnv,
    SharedMemHumanoidCollabVecEnv,
    SubprocHumanoidCollabVecEnv,
)
from humanoid_collab.amp.amp_obs import AMPObsBuilder, quat_to_mat
from humanoid_collab.amp.discriminator import AMPDiscriminator, AMPDiscriminatorTrainer
from humanoid_collab.amp.motion_buffer import MotionReplayBuffer, PolicyTransitionBuffer
from humanoid_collab.amp.motion_data import load_motion_dataset
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.vision import load_frozen_encoder_from_vae_checkpoint
from humanoid_collab.utils.exp_logging import ExperimentLogger


AGENTS = ("h0", "h1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MADDPG on HumanoidCollabEnv.")

    parser.add_argument("--task", type=str, default="handshake", choices=["hug", "handshake", "box_lift"])
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--physics-profile", type=str, default="default", choices=available_physics_profiles())
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=30)
    parser.add_argument(
        "--fixed-standing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Weld torsos to world (fixed standing). Use --no-fixed-standing for locomotion.",
    )
    parser.add_argument("--control-mode", type=str, default="arms_only", choices=["all", "arms_only"])
    parser.add_argument("--observation-mode", type=str, default="proprio", choices=["proprio", "rgb", "gray"])
    parser.add_argument("--obs-rgb-width", type=int, default=84)
    parser.add_argument("--obs-rgb-height", type=int, default=84)

    parser.add_argument("--total-steps", type=int, default=800_000, help="Total environment transitions.")
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--start-steps", type=int, default=8_000)
    parser.add_argument("--update-after", type=int, default=2_000)
    parser.add_argument(
        "--update-every",
        type=int,
        default=1,
        help="Run learning updates every N collector iterations.",
    )
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
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained VAE checkpoint for visual observations.",
    )
    parser.add_argument(
        "--vision-use-proprio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate proprio vectors from infos with visual latents (recommended).",
    )
    parser.add_argument(
        "--replay-obs-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Storage dtype for encoded observations in replay buffer.",
    )

    parser.add_argument("--auto-curriculum", action="store_true", help="Advance stage based on success window.")
    parser.add_argument(
        "--curriculum-window-episodes",
        type=int,
        default=60,
        help="Episode window used to evaluate stage promotion.",
    )
    parser.add_argument(
        "--curriculum-success-threshold",
        type=float,
        default=0.65,
        help="Default success-rate threshold for stage promotion.",
    )
    parser.add_argument(
        "--curriculum-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional per-promotion success thresholds from starting stage upward.",
    )
    parser.add_argument(
        "--curriculum-min-episodes-per-stage",
        type=int,
        default=80,
        help="Minimum completed episodes in a stage before promotion.",
    )

    parser.add_argument("--log-dir", type=str, default="runs/maddpg_handshake_fixed_arms")
    parser.add_argument("--save-dir", type=str, default="checkpoints/maddpg_handshake_fixed_arms")
    parser.add_argument("--save-every-steps", type=int, default=50_000)
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a checkpoint path.")
    parser.add_argument("--print-every-steps", type=int, default=2_048)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="maddpg")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--amp-enable", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--amp-warmup-steps", type=int, default=50_000)
    parser.add_argument("--amp-policy-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--amp-freeze-discriminator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-reward-scale", type=float, default=2.0)
    parser.add_argument("--amp-clip-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-approach-steps", type=int, default=500_000)
    parser.add_argument("--amp-weight-override", type=float, default=None)
    parser.add_argument("--amp-weight-stage0", type=float, default=0.40)
    parser.add_argument("--amp-weight-stage1", type=float, default=0.35)
    parser.add_argument("--amp-weight-stage2", type=float, default=0.30)
    parser.add_argument("--amp-weight-stage3", type=float, default=0.25)
    parser.add_argument("--amp-include-root-height", action=argparse.BooleanOptionalAction, default=True)

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
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256):
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
    def __init__(self, input_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, joint_obs_act: torch.Tensor) -> torch.Tensor:
        return self.net(joint_obs_act)


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        obs_dtype: np.dtype = np.float32,
    ):
        if capacity <= 0:
            raise ValueError("Replay capacity must be > 0")
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.obs_dtype = np.dtype(obs_dtype)

        self.obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=self.obs_dtype)
        self.actions = np.zeros((self.capacity, self.num_agents, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=self.obs_dtype)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        batch_n = int(obs.shape[0])
        obs_arr = np.asarray(obs, dtype=self.obs_dtype)
        next_obs_arr = np.asarray(next_obs, dtype=self.obs_dtype)
        for i in range(batch_n):
            idx = self.ptr
            self.obs[idx] = obs_arr[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_obs[idx] = next_obs_arr[i]
            self.dones[idx, 0] = float(dones[i, 0])
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


@dataclass
class ObsAdapter:
    use_visual_encoder: bool
    use_proprio_with_visual: bool
    encoder_bundle: Optional[Any]
    encoder_device: torch.device
    expected_obs_shape: Optional[Tuple[int, int, int]]


def _build_obs_adapter(args: argparse.Namespace, device: torch.device) -> ObsAdapter:
    if args.observation_mode == "proprio":
        return ObsAdapter(
            use_visual_encoder=False,
            use_proprio_with_visual=False,
            encoder_bundle=None,
            encoder_device=device,
            expected_obs_shape=None,
        )
    if args.encoder_checkpoint is None:
        raise ValueError(
            "Visual MADDPG requires --encoder-checkpoint pointing to a pretrained VAE."
        )
    bundle = load_frozen_encoder_from_vae_checkpoint(args.encoder_checkpoint, device=device)
    if bundle.observation_mode != args.observation_mode:
        raise ValueError(
            f"Encoder observation_mode='{bundle.observation_mode}' does not match "
            f"--observation-mode '{args.observation_mode}'."
        )
    return ObsAdapter(
        use_visual_encoder=True,
        use_proprio_with_visual=bool(args.vision_use_proprio),
        encoder_bundle=bundle,
        encoder_device=device,
        expected_obs_shape=bundle.obs_shape,
    )


def _normalize_infos_payload(
    infos_payload: Any,
    num_envs: int,
) -> List[Dict[str, Dict[str, Any]]]:
    if num_envs == 1 and isinstance(infos_payload, dict):
        return [infos_payload]
    if not isinstance(infos_payload, list):
        raise TypeError(f"Expected list infos payload for num_envs={num_envs}, got {type(infos_payload)}")
    return infos_payload


def _extract_proprio_from_infos(
    infos_list: List[Dict[str, Dict[str, Any]]],
    num_envs: int,
    done_mask: Optional[np.ndarray],
    use_reset_info_for_done: bool,
) -> Dict[str, np.ndarray]:
    if done_mask is None:
        done_mask = np.zeros((num_envs,), dtype=np.bool_)
    out: Dict[str, List[np.ndarray]] = {agent: [] for agent in AGENTS}
    for env_idx in range(num_envs):
        info_env = infos_list[env_idx]
        done_flag = bool(done_mask[env_idx])
        for agent in AGENTS:
            agent_info = info_env.get(agent, {})
            info_src = agent_info
            if use_reset_info_for_done and done_flag:
                reset_info = agent_info.get("reset_info")
                if isinstance(reset_info, dict):
                    info_src = reset_info
            vec = info_src.get("proprio_obs")
            if vec is None:
                raise KeyError(
                    "Expected 'proprio_obs' in infos for visual training. "
                    "Ensure env was created with emit_proprio_info=True."
                )
            out[agent].append(np.asarray(vec, dtype=np.float32))
    return {agent: np.stack(out[agent], axis=0).astype(np.float32) for agent in AGENTS}


def _encode_visual_batch(
    raw_obs_batch: Dict[str, np.ndarray],
    adapter: ObsAdapter,
) -> Dict[str, np.ndarray]:
    assert adapter.encoder_bundle is not None
    expected_shape = adapter.expected_obs_shape
    latent: Dict[str, np.ndarray] = {}
    for agent in AGENTS:
        obs_np = np.asarray(raw_obs_batch[agent])
        if obs_np.ndim != 4:
            raise ValueError(
                f"Expected batched image observations [N,H,W,C] for agent '{agent}', got {obs_np.shape}"
            )
        h, w, c = int(obs_np.shape[1]), int(obs_np.shape[2]), int(obs_np.shape[3])
        if expected_shape is not None and (h, w, c) != expected_shape:
            raise ValueError(
                f"Encoder expects observation shape {expected_shape}, got {(h, w, c)} for agent '{agent}'."
            )
        x = torch.as_tensor(obs_np, device=adapter.encoder_device, dtype=torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            z = adapter.encoder_bundle.vae.encode_mean(x)
        latent[agent] = z.detach().cpu().numpy().astype(np.float32)
    return latent


def _transform_obs_batch(
    raw_obs_batch: Dict[str, np.ndarray],
    infos_payload: Any,
    num_envs: int,
    adapter: ObsAdapter,
    done_mask: Optional[np.ndarray] = None,
    use_reset_info_for_done: bool = False,
) -> Dict[str, np.ndarray]:
    if not adapter.use_visual_encoder:
        return {agent: np.asarray(raw_obs_batch[agent], dtype=np.float32) for agent in AGENTS}
    infos_list = _normalize_infos_payload(infos_payload, num_envs)
    latent = _encode_visual_batch(raw_obs_batch, adapter)
    if not adapter.use_proprio_with_visual:
        return latent
    proprio = _extract_proprio_from_infos(
        infos_list=infos_list,
        num_envs=num_envs,
        done_mask=done_mask,
        use_reset_info_for_done=use_reset_info_for_done,
    )
    return {
        agent: np.concatenate([proprio[agent], latent[agent]], axis=-1).astype(np.float32)
        for agent in AGENTS
    }


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau)
            p_t.data.add_(tau * p.data)


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


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


def _mean_recent(values: List[float], n: int) -> float:
    if not values:
        return 0.0
    k = min(len(values), max(1, n))
    return float(np.mean(values[-k:]))


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


def _extract_root_height_from_infos(
    infos_list: List[Dict[str, Dict[str, Any]]],
    num_envs: int,
    done_mask: Optional[np.ndarray],
    use_reset_info_for_done: bool,
) -> Dict[str, np.ndarray]:
    if done_mask is None:
        done_mask = np.zeros((num_envs,), dtype=np.bool_)
    out: Dict[str, List[float]] = {agent: [] for agent in AGENTS}
    for env_idx in range(num_envs):
        info_env = infos_list[env_idx]
        done_flag = bool(done_mask[env_idx])
        for agent in AGENTS:
            agent_info = info_env.get(agent, {})
            info_src = agent_info
            if use_reset_info_for_done and done_flag:
                reset_info = agent_info.get("reset_info")
                if isinstance(reset_info, dict):
                    info_src = reset_info
            if "root_height" not in info_src:
                raise KeyError("Expected 'root_height' in infos payload for AMP training.")
            out[agent].append(float(info_src["root_height"]))
    return {
        agent: np.asarray(values, dtype=np.float32)
        for agent, values in out.items()
    }


def _extract_proprio_for_amp(
    raw_obs_batch: Dict[str, np.ndarray],
    infos_payload: Any,
    num_envs: int,
    observation_mode: str,
    done_mask: Optional[np.ndarray],
    use_reset_info_for_done: bool,
) -> Dict[str, np.ndarray]:
    if observation_mode == "proprio":
        return {agent: np.asarray(raw_obs_batch[agent], dtype=np.float32) for agent in AGENTS}
    infos_list = _normalize_infos_payload(infos_payload, num_envs)
    return _extract_proprio_from_infos(
        infos_list=infos_list,
        num_envs=num_envs,
        done_mask=done_mask,
        use_reset_info_for_done=use_reset_info_for_done,
    )


def _build_amp_obs_from_proprio(
    proprio_batch: Dict[str, np.ndarray],
    root_height_batch: Dict[str, np.ndarray],
    joint_qpos_dim: int,
    joint_qvel_dim: int,
    include_root_height: bool,
) -> Dict[str, np.ndarray]:
    base_dim = joint_qpos_dim + joint_qvel_dim + 4 + 3
    out: Dict[str, np.ndarray] = {}
    for agent in AGENTS:
        proprio = np.asarray(proprio_batch[agent], dtype=np.float32)
        if proprio.ndim != 2:
            raise ValueError(f"Expected 2D proprio batch for {agent}, got {proprio.shape}")
        if proprio.shape[1] < base_dim:
            raise ValueError(
                f"Proprio dim for {agent} is {proprio.shape[1]}, expected at least {base_dim}."
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
            parts.append(root_height_batch[agent].reshape(-1, 1))
        parts.extend([fwd, up, root_angvel])
        out[agent] = np.concatenate(parts, axis=-1).astype(np.float32)
    return out


def _compute_amp_obs_batch(
    *,
    raw_obs_batch: Dict[str, np.ndarray],
    infos_payload: Any,
    num_envs: int,
    observation_mode: str,
    done_mask: Optional[np.ndarray],
    use_reset_info_for_done: bool,
    joint_qpos_dim: int,
    joint_qvel_dim: int,
    include_root_height: bool,
) -> Dict[str, np.ndarray]:
    infos_list = _normalize_infos_payload(infos_payload, num_envs)
    proprio = _extract_proprio_for_amp(
        raw_obs_batch=raw_obs_batch,
        infos_payload=infos_payload,
        num_envs=num_envs,
        observation_mode=observation_mode,
        done_mask=done_mask,
        use_reset_info_for_done=use_reset_info_for_done,
    )
    root_height = _extract_root_height_from_infos(
        infos_list=infos_list,
        num_envs=num_envs,
        done_mask=done_mask,
        use_reset_info_for_done=use_reset_info_for_done,
    )
    return _build_amp_obs_from_proprio(
        proprio_batch=proprio,
        root_height_batch=root_height,
        joint_qpos_dim=joint_qpos_dim,
        joint_qvel_dim=joint_qvel_dim,
        include_root_height=include_root_height,
    )


def _make_env_kwargs(args: argparse.Namespace, stage: int) -> Dict[str, object]:
    emit_proprio_info = bool(
        args.observation_mode in {"rgb", "gray"}
        and (args.vision_use_proprio or args.amp_enable)
    )
    return dict(
        task=args.task,
        stage=int(stage),
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        hold_target=args.hold_target,
        physics_profile=args.physics_profile,
        fixed_standing=args.fixed_standing,
        control_mode=args.control_mode,
        observation_mode=args.observation_mode,
        obs_rgb_width=int(args.obs_rgb_width),
        obs_rgb_height=int(args.obs_rgb_height),
        emit_proprio_info=emit_proprio_info,
    )


def _format_reset_obs(obs: Dict[str, np.ndarray], num_envs: int) -> Dict[str, np.ndarray]:
    if num_envs == 1:
        return {
            agent: np.asarray(obs[agent])[None, ...]
            for agent in AGENTS
        }
    return {
        agent: np.asarray(obs[agent])
        for agent in AGENTS
    }


def _build_env(
    args: argparse.Namespace,
    stage: int,
    seed: Optional[int],
):
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
    obs_batch = _format_reset_obs(obs, args.num_envs)
    return env, obs_batch, infos


def _collect_actions(
    obs_batch: Dict[str, np.ndarray],
    actors: Dict[str, Actor],
    act_dim: int,
    start_steps_phase: bool,
    noise_std: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    num_envs = obs_batch[AGENTS[0]].shape[0]
    actions: Dict[str, np.ndarray] = {}

    if start_steps_phase:
        for agent in AGENTS:
            actions[agent] = np.random.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
        return actions

    for agent in AGENTS:
        obs_t = torch.as_tensor(obs_batch[agent], dtype=torch.float32, device=device)
        with torch.no_grad():
            act = actors[agent](obs_t).cpu().numpy()
        if noise_std > 0.0:
            act = act + np.random.normal(0.0, noise_std, size=act.shape)
        actions[agent] = np.clip(act, -1.0, 1.0).astype(np.float32)

    return actions


def _stack_joint(
    obs_batch: Dict[str, np.ndarray],
    actions_batch: Dict[str, np.ndarray],
    rewards_batch: Dict[str, np.ndarray],
    next_obs_batch: Dict[str, np.ndarray],
    done_batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.stack([obs_batch[agent] for agent in AGENTS], axis=1).astype(np.float32)
    actions = np.stack([actions_batch[agent] for agent in AGENTS], axis=1).astype(np.float32)
    rewards = np.stack([rewards_batch[agent] for agent in AGENTS], axis=1).astype(np.float32)
    next_obs = np.stack([next_obs_batch[agent] for agent in AGENTS], axis=1).astype(np.float32)
    dones = done_batch.astype(np.float32).reshape(-1, 1)
    return obs, actions, rewards, next_obs, dones


def _extract_done_and_reasons_single(
    terminations: Dict[str, bool],
    truncations: Dict[str, bool],
    infos: Dict[str, Dict[str, object]],
) -> Tuple[np.ndarray, List[str]]:
    done = bool(terminations["h0"] or truncations["h0"])
    reason = str(infos["h0"].get("termination_reason", "running"))
    return np.asarray([done], dtype=np.bool_), [reason]


def _extract_done_and_reasons_vec(
    terminations: Dict[str, np.ndarray],
    truncations: Dict[str, np.ndarray],
    infos: List[Dict[str, Dict[str, object]]],
) -> Tuple[np.ndarray, List[str]]:
    done = np.logical_or(terminations["h0"], truncations["h0"])
    reasons: List[str] = []
    for i in range(done.shape[0]):
        if done[i]:
            reasons.append(str(infos[i]["h0"].get("termination_reason", "unknown")))
        else:
            reasons.append("running")
    return done.astype(np.bool_), reasons


def _maybe_replace_final_obs_from_infos(
    next_obs_batch: Dict[str, np.ndarray],
    done_mask: np.ndarray,
    infos: List[Dict[str, Dict[str, object]]],
) -> Dict[str, np.ndarray]:
    out = {agent: np.asarray(next_obs_batch[agent]).copy() for agent in AGENTS}
    for env_idx, done in enumerate(done_mask):
        if not bool(done):
            continue
        info_env = infos[env_idx]
        for agent in AGENTS:
            final_obs = info_env[agent].get("final_observation")
            if final_obs is not None:
                out[agent][env_idx] = np.asarray(final_obs)
    return out


def train(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.buffer_size <= args.batch_size:
        raise ValueError("--buffer-size should be larger than --batch-size")
    if args.curriculum_window_episodes <= 0:
        raise ValueError("--curriculum-window-episodes must be > 0")
    if args.curriculum_min_episodes_per_stage <= 0:
        raise ValueError("--curriculum-min-episodes-per-stage must be > 0")
    if args.exploration_noise_start < 0.0 or args.exploration_noise_end < 0.0:
        raise ValueError("exploration noise values must be non-negative")
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0")
    if args.amp_enable and args.task != "hug":
        raise ValueError("--amp-enable currently supports --task hug only.")
    if args.amp_enable and not args.amp_motion_data_dir:
        raise ValueError("--amp-enable requires --amp-motion-data-dir.")

    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    probe_env = HumanoidCollabEnv(**_make_env_kwargs(args, args.stage))
    max_stage = int(probe_env.task_config.num_curriculum_stages) - 1
    if max_stage < 0:
        max_stage = 0
    probe_env.close()

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

    env, raw_obs_batch, reset_infos = _build_env(args, stage=current_stage, seed=args.seed)
    obs_adapter = _build_obs_adapter(args, device)
    obs_batch = _transform_obs_batch(
        raw_obs_batch=raw_obs_batch,
        infos_payload=reset_infos,
        num_envs=args.num_envs,
        adapter=obs_adapter,
    )

    if obs_batch["h0"].ndim != 2:
        env.close()
        raise ValueError(
            f"Expected 2D encoded observations [N,obs_dim], got shape {obs_batch['h0'].shape}."
        )

    obs_dim = int(obs_batch["h0"].shape[-1])
    act_dim = int(env.action_space("h0").shape[0])
    n_agents = len(AGENTS)
    joint_input_dim = n_agents * (obs_dim + act_dim)

    amp_motion_buffer: Optional[MotionReplayBuffer] = None
    amp_policy_buffer: Optional[PolicyTransitionBuffer] = None
    amp_discriminator: Optional[AMPDiscriminator] = None
    amp_disc_trainer: Optional[AMPDiscriminatorTrainer] = None
    amp_joint_qpos_dim = 0
    amp_joint_qvel_dim = 0
    amp_obs_dim = 0

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
            tmp_kwargs = _make_env_kwargs(args, current_stage)
            tmp_kwargs["observation_mode"] = "proprio"
            tmp_kwargs["emit_proprio_info"] = False
            tmp_env = HumanoidCollabEnv(**tmp_kwargs)
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
        )
        amp_discriminator.to(device)
        if not args.amp_freeze_discriminator:
            amp_disc_trainer = AMPDiscriminatorTrainer(
                discriminator=amp_discriminator,
                lr=args.amp_disc_lr,
                lambda_gp=args.amp_lambda_gp,
                n_updates=args.amp_disc_updates,
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

    actors: Dict[str, Actor] = {}
    critics: Dict[str, Critic] = {}
    target_actors: Dict[str, Actor] = {}
    target_critics: Dict[str, Critic] = {}
    actor_optim: Dict[str, optim.Optimizer] = {}
    critic_optim: Dict[str, optim.Optimizer] = {}

    for agent in AGENTS:
        actor = Actor(obs_dim, act_dim, args.hidden_size).to(device)
        critic = Critic(joint_input_dim, args.hidden_size).to(device)
        target_actor = Actor(obs_dim, act_dim, args.hidden_size).to(device)
        target_critic = Critic(joint_input_dim, args.hidden_size).to(device)

        _hard_update(target_actor, actor)
        _hard_update(target_critic, critic)

        actors[agent] = actor
        critics[agent] = critic
        target_actors[agent] = target_actor
        target_critics[agent] = target_critic
        actor_optim[agent] = optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_optim[agent] = optim.Adam(critic.parameters(), lr=args.critic_lr)

    replay = ReplayBuffer(
        capacity=args.buffer_size,
        num_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_dtype=np.float16 if args.replay_obs_dtype == "float16" else np.float32,
    )

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

    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)
    completed_returns: List[float] = []
    completed_lengths: List[float] = []
    completed_successes: List[float] = []
    episodes_in_stage = 0

    start_time = time.time()
    last_print_step = 0
    last_save_step = 0
    global_step = 0
    collector_iter = 0
    grad_updates = 0
    actor_updates = 0

    # rolling metrics from learning updates
    critic_loss_hist = {agent: [] for agent in AGENTS}
    actor_loss_hist = {agent: [] for agent in AGENTS}
    q_hist = {agent: [] for agent in AGENTS}
    hug_contact_any_hist: List[float] = []
    hug_dual_contact_hist: List[float] = []
    hug_hand_back_mean_hist: List[float] = []
    amp_reward_hist: List[float] = []
    amp_weight_hist: List[float] = []
    amp_disc_loss_hist: List[float] = []
    amp_d_real_hist: List[float] = []
    amp_d_fake_hist: List[float] = []

    amp_obs_current: Optional[Dict[str, np.ndarray]] = None
    if args.amp_enable:
        amp_obs_current = _compute_amp_obs_batch(
            raw_obs_batch=raw_obs_batch,
            infos_payload=reset_infos,
            num_envs=args.num_envs,
            observation_mode=args.observation_mode,
            done_mask=None,
            use_reset_info_for_done=False,
            joint_qpos_dim=amp_joint_qpos_dim,
            joint_qvel_dim=amp_joint_qvel_dim,
            include_root_height=bool(args.amp_include_root_height),
        )

    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            env.close()
            logger.finish()
            raise FileNotFoundError(f"--resume-from checkpoint not found: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)

        ckpt_obs_dim = int(ckpt.get("obs_dim", obs_dim))
        ckpt_act_dim = int(ckpt.get("act_dim", act_dim))
        ckpt_agents = int(ckpt.get("n_agents", n_agents))
        if ckpt_obs_dim != obs_dim or ckpt_act_dim != act_dim or ckpt_agents != n_agents:
            env.close()
            logger.finish()
            raise ValueError(
                "Resume checkpoint dimensions do not match current run config: "
                f"ckpt(obs={ckpt_obs_dim}, act={ckpt_act_dim}, n_agents={ckpt_agents}) "
                f"!= run(obs={obs_dim}, act={act_dim}, n_agents={n_agents})."
            )
        if "actors" not in ckpt or "critics" not in ckpt:
            env.close()
            logger.finish()
            raise KeyError(
                f"Checkpoint at '{args.resume_from}' is missing actor/critic weights."
            )

        for agent in AGENTS:
            actors[agent].load_state_dict(ckpt["actors"][agent])
            critics[agent].load_state_dict(ckpt["critics"][agent])
            if "target_actors" in ckpt and agent in ckpt["target_actors"]:
                target_actors[agent].load_state_dict(ckpt["target_actors"][agent])
            else:
                _hard_update(target_actors[agent], actors[agent])
            if "target_critics" in ckpt and agent in ckpt["target_critics"]:
                target_critics[agent].load_state_dict(ckpt["target_critics"][agent])
            else:
                _hard_update(target_critics[agent], critics[agent])
            if "actor_optim" in ckpt and agent in ckpt["actor_optim"]:
                actor_optim[agent].load_state_dict(ckpt["actor_optim"][agent])
            if "critic_optim" in ckpt and agent in ckpt["critic_optim"]:
                critic_optim[agent].load_state_dict(ckpt["critic_optim"][agent])

        if args.amp_enable and amp_discriminator is not None:
            if "amp_discriminator" in ckpt:
                amp_discriminator.load_state_dict(ckpt["amp_discriminator"])
            if (
                amp_disc_trainer is not None
                and "amp_disc_optimizer" in ckpt
                and ckpt["amp_disc_optimizer"] is not None
            ):
                amp_disc_trainer.optimizer.load_state_dict(ckpt["amp_disc_optimizer"])

        global_step = int(ckpt.get("global_step", 0))
        collector_iter = int(ckpt.get("collector_iter", 0))
        grad_updates = int(ckpt.get("grad_updates", 0))
        actor_updates = int(ckpt.get("actor_updates", 0))
        last_print_step = int(ckpt.get("last_print_step", global_step))
        last_save_step = int(ckpt.get("last_save_step", global_step))
        episodes_in_stage = int(ckpt.get("episodes_in_stage", episodes_in_stage))

        current_stage = int(np.clip(int(ckpt.get("stage", current_stage)), 0, max_stage))
        env.close()
        env, raw_obs_batch, reset_infos = _build_env(args, stage=current_stage, seed=None)
        obs_batch = _transform_obs_batch(
            raw_obs_batch=raw_obs_batch,
            infos_payload=reset_infos,
            num_envs=args.num_envs,
            adapter=obs_adapter,
        )
        if args.amp_enable:
            amp_obs_current = _compute_amp_obs_batch(
                raw_obs_batch=raw_obs_batch,
                infos_payload=reset_infos,
                num_envs=args.num_envs,
                observation_mode=args.observation_mode,
                done_mask=None,
                use_reset_info_for_done=False,
                joint_qpos_dim=amp_joint_qpos_dim,
                joint_qvel_dim=amp_joint_qvel_dim,
                include_root_height=bool(args.amp_include_root_height),
            )
        print(
            f"resumed from {args.resume_from}: "
            f"global_step={global_step} stage={current_stage} "
            f"grad_updates={grad_updates} actor_updates={actor_updates}"
        )

    def _noise_std(step_count: int) -> float:
        if step_count <= args.start_steps:
            return float(args.exploration_noise_start)
        denom = max(1, args.total_steps - args.start_steps)
        frac = min(1.0, max(0.0, (step_count - args.start_steps) / denom))
        return float(
            args.exploration_noise_start
            + frac * (args.exploration_noise_end - args.exploration_noise_start)
        )

    try:
        while global_step < args.total_steps:
            collector_iter += 1
            current_noise = _noise_std(global_step)
            amp_disc_metrics = {"disc_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
            actions_batch = _collect_actions(
                obs_batch=obs_batch,
                actors=actors,
                act_dim=act_dim,
                start_steps_phase=(global_step < args.start_steps),
                noise_std=current_noise,
                device=device,
            )

            if args.num_envs == 1:
                env_actions = {agent: actions_batch[agent][0] for agent in AGENTS}
                next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(env_actions)

                done_mask, reasons = _extract_done_and_reasons_single(terminations, truncations, infos)
                raw_next_obs_for_buffer = {
                    agent: np.asarray(next_obs_dict[agent])[None, ...]
                    for agent in AGENTS
                }
                raw_next_obs_for_rollout = raw_next_obs_for_buffer
                infos_list = [infos]
                rollout_infos_payload: Any = infos_list

                rewards_batch = {
                    agent: np.asarray([float(rewards_dict[agent])], dtype=np.float32)
                    for agent in AGENTS
                }

                if bool(done_mask[0]):
                    reset_obs, reset_infos = env.reset(seed=None, options={"stage": int(current_stage)})
                    raw_next_obs_for_rollout = {
                        agent: np.asarray(reset_obs[agent])[None, ...]
                        for agent in AGENTS
                    }
                    rollout_infos_payload = reset_infos

            else:
                next_obs_vec, rewards_vec, terminations, truncations, infos_list = env.step(actions_batch)
                done_mask, reasons = _extract_done_and_reasons_vec(terminations, truncations, infos_list)

                raw_next_obs_for_rollout = {
                    agent: np.asarray(next_obs_vec[agent])
                    for agent in AGENTS
                }
                raw_next_obs_for_buffer = _maybe_replace_final_obs_from_infos(
                    next_obs_batch=raw_next_obs_for_rollout,
                    done_mask=done_mask,
                    infos=infos_list,
                )
                rewards_batch = {
                    agent: np.asarray(rewards_vec[agent], dtype=np.float32)
                    for agent in AGENTS
                }
                rollout_infos_payload = infos_list

            if args.amp_enable:
                if amp_obs_current is None or amp_discriminator is None or amp_policy_buffer is None:
                    raise RuntimeError("AMP is enabled but AMP components are not initialized.")

                amp_obs_t = amp_obs_current
                amp_obs_t1_for_buffer = _compute_amp_obs_batch(
                    raw_obs_batch=raw_next_obs_for_buffer,
                    infos_payload=infos_list,
                    num_envs=args.num_envs,
                    observation_mode=args.observation_mode,
                    done_mask=done_mask,
                    use_reset_info_for_done=False,
                    joint_qpos_dim=amp_joint_qpos_dim,
                    joint_qvel_dim=amp_joint_qvel_dim,
                    include_root_height=bool(args.amp_include_root_height),
                )

                amp_policy_buffer.add_batch(
                    np.concatenate([amp_obs_t["h0"], amp_obs_t["h1"]], axis=0),
                    np.concatenate([amp_obs_t1_for_buffer["h0"], amp_obs_t1_for_buffer["h1"]], axis=0),
                )

                amp_style_rewards: Dict[str, np.ndarray] = {}
                for agent in AGENTS:
                    obs_t_t = torch.as_tensor(amp_obs_t[agent], dtype=torch.float32, device=device)
                    obs_t1_t = torch.as_tensor(amp_obs_t1_for_buffer[agent], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        r_style = amp_discriminator.compute_reward(
                            obs_t_t,
                            obs_t1_t,
                            clip_reward=bool(args.amp_clip_reward),
                            reward_scale=float(args.amp_reward_scale),
                        )
                    amp_style_rewards[agent] = r_style.detach().cpu().numpy().astype(np.float32)

                in_approach_phase = bool(global_step < args.amp_approach_steps)
                amp_weight = 0.5 if in_approach_phase else _amp_weight_for_stage(args, current_stage)
                task_weight = 1.0 - amp_weight
                amp_weight_hist.append(float(amp_weight))

                for agent in AGENTS:
                    if in_approach_phase:
                        approach_vals = []
                        for info_env in infos_list:
                            agent_info = info_env.get(agent, {})
                            approach_vals.append(
                                float(agent_info.get("r_distance", 0.0))
                                + float(agent_info.get("r_facing", 0.0))
                            )
                        task_component = np.asarray(approach_vals, dtype=np.float32)
                    else:
                        task_component = np.asarray(rewards_batch[agent], dtype=np.float32)

                    rewards_batch[agent] = (
                        amp_weight * amp_style_rewards[agent] + task_weight * task_component
                    ).astype(np.float32)
                    amp_reward_hist.extend(amp_style_rewards[agent].tolist())

                max_amp_hist = 20_000
                if len(amp_reward_hist) > max_amp_hist:
                    del amp_reward_hist[:-max_amp_hist]
                if len(amp_weight_hist) > max_amp_hist:
                    del amp_weight_hist[:-max_amp_hist]

                amp_obs_current = _compute_amp_obs_batch(
                    raw_obs_batch=raw_next_obs_for_rollout,
                    infos_payload=rollout_infos_payload,
                    num_envs=args.num_envs,
                    observation_mode=args.observation_mode,
                    done_mask=done_mask,
                    use_reset_info_for_done=(args.num_envs > 1),
                    joint_qpos_dim=amp_joint_qpos_dim,
                    joint_qvel_dim=amp_joint_qvel_dim,
                    include_root_height=bool(args.amp_include_root_height),
                )

            if args.task == "hug":
                for info_env in infos_list:
                    h0_info = info_env.get("h0", {})
                    if not isinstance(h0_info, dict):
                        continue
                    if "hug_contact_any" in h0_info:
                        hug_contact_any_hist.append(float(h0_info["hug_contact_any"]))
                    if "hug_dual_contact" in h0_info:
                        hug_dual_contact_hist.append(float(h0_info["hug_dual_contact"]))
                    if "hug_hand_back_mean" in h0_info:
                        hug_hand_back_mean_hist.append(float(h0_info["hug_hand_back_mean"]))
                max_hist = 20_000
                if len(hug_contact_any_hist) > max_hist:
                    del hug_contact_any_hist[:-max_hist]
                if len(hug_dual_contact_hist) > max_hist:
                    del hug_dual_contact_hist[:-max_hist]
                if len(hug_hand_back_mean_hist) > max_hist:
                    del hug_hand_back_mean_hist[:-max_hist]

            next_obs_for_buffer = _transform_obs_batch(
                raw_obs_batch=raw_next_obs_for_buffer,
                infos_payload=infos_list,
                num_envs=args.num_envs,
                adapter=obs_adapter,
                done_mask=done_mask,
                use_reset_info_for_done=False,
            )
            next_obs_for_rollout = _transform_obs_batch(
                raw_obs_batch=raw_next_obs_for_rollout,
                infos_payload=rollout_infos_payload,
                num_envs=args.num_envs,
                adapter=obs_adapter,
                done_mask=done_mask,
                use_reset_info_for_done=(args.num_envs > 1),
            )

            obs_joint, act_joint, rew_joint, next_obs_joint, done_joint = _stack_joint(
                obs_batch=obs_batch,
                actions_batch=actions_batch,
                rewards_batch=rewards_batch,
                next_obs_batch=next_obs_for_buffer,
                done_batch=done_mask,
            )
            replay.add_batch(obs_joint, act_joint, rew_joint, next_obs_joint, done_joint)

            ep_returns += rewards_batch["h0"]
            ep_lengths += 1
            for env_idx, done in enumerate(done_mask):
                if not bool(done):
                    continue
                completed_returns.append(float(ep_returns[env_idx]))
                completed_lengths.append(float(ep_lengths[env_idx]))
                episodes_in_stage += 1
                completed_successes.append(1.0 if reasons[env_idx] == "success" else 0.0)
                ep_returns[env_idx] = 0.0
                ep_lengths[env_idx] = 0

            obs_batch = next_obs_for_rollout
            global_step += args.num_envs

            if (
                args.amp_enable
                and amp_disc_trainer is not None
                and amp_motion_buffer is not None
                and amp_policy_buffer is not None
                and global_step >= args.amp_warmup_steps
                and amp_policy_buffer.size >= args.amp_disc_batch_size
            ):
                real_obs_t, real_obs_t1 = amp_motion_buffer.sample_torch(
                    args.amp_disc_batch_size,
                    device=str(device),
                )
                fake_obs_t, fake_obs_t1 = amp_policy_buffer.sample_torch(
                    args.amp_disc_batch_size,
                    device=str(device),
                )
                amp_disc_metrics = amp_disc_trainer.train_step(
                    real_obs_t,
                    real_obs_t1,
                    fake_obs_t,
                    fake_obs_t1,
                )
                amp_disc_loss_hist.append(float(amp_disc_metrics["disc_loss"]))
                amp_d_real_hist.append(float(amp_disc_metrics["d_real"]))
                amp_d_fake_hist.append(float(amp_disc_metrics["d_fake"]))
                max_disc_hist = 2_000
                if len(amp_disc_loss_hist) > max_disc_hist:
                    del amp_disc_loss_hist[:-max_disc_hist]
                if len(amp_d_real_hist) > max_disc_hist:
                    del amp_d_real_hist[:-max_disc_hist]
                if len(amp_d_fake_hist) > max_disc_hist:
                    del amp_d_fake_hist[:-max_disc_hist]

            if (
                global_step >= args.update_after
                and replay.size >= args.batch_size
                and collector_iter % args.update_every == 0
            ):
                for _ in range(args.gradient_steps):
                    batch = replay.sample(args.batch_size, device=device)
                    bsz = batch.obs.shape[0]

                    obs_flat = batch.obs.reshape(bsz, -1)
                    next_obs_flat = batch.next_obs.reshape(bsz, -1)
                    act_flat = batch.actions.reshape(bsz, -1)

                    with torch.no_grad():
                        next_actions = []
                        for agent_idx, agent in enumerate(AGENTS):
                            next_actions.append(target_actors[agent](batch.next_obs[:, agent_idx, :]))
                        next_act_flat = torch.cat(next_actions, dim=-1)
                        next_joint = torch.cat([next_obs_flat, next_act_flat], dim=-1)

                    for agent_idx, agent in enumerate(AGENTS):
                        q_in = torch.cat([obs_flat, act_flat], dim=-1)
                        q_pred = critics[agent](q_in)

                        with torch.no_grad():
                            q_next = target_critics[agent](next_joint)
                            y = batch.rewards[:, agent_idx : agent_idx + 1] + args.gamma * (
                                1.0 - batch.dones
                            ) * q_next

                        c_loss = nn.functional.mse_loss(q_pred, y)
                        critic_optim[agent].zero_grad(set_to_none=True)
                        c_loss.backward()
                        nn.utils.clip_grad_norm_(critics[agent].parameters(), args.max_grad_norm)
                        critic_optim[agent].step()

                        critic_loss_hist[agent].append(float(c_loss.item()))
                        q_hist[agent].append(float(q_pred.mean().item()))

                    grad_updates += 1

                    if grad_updates % args.policy_delay == 0:
                        for agent_idx, agent in enumerate(AGENTS):
                            for p in critics[agent].parameters():
                                p.requires_grad_(False)
                            policy_actions = []
                            for other_idx, other_agent in enumerate(AGENTS):
                                a_pred = actors[other_agent](batch.obs[:, other_idx, :])
                                if other_idx != agent_idx:
                                    a_pred = a_pred.detach()
                                policy_actions.append(a_pred)
                            policy_act_flat = torch.cat(policy_actions, dim=-1)
                            policy_q_in = torch.cat([obs_flat, policy_act_flat], dim=-1)
                            policy_q = critics[agent](policy_q_in)

                            a_loss = -policy_q.mean()
                            if args.action_l2 > 0.0:
                                a_loss = a_loss + args.action_l2 * (policy_actions[agent_idx] ** 2).mean()

                            actor_optim[agent].zero_grad(set_to_none=True)
                            a_loss.backward()
                            nn.utils.clip_grad_norm_(actors[agent].parameters(), args.max_grad_norm)
                            actor_optim[agent].step()
                            for p in critics[agent].parameters():
                                p.requires_grad_(True)

                            actor_loss_hist[agent].append(float(a_loss.item()))

                        actor_updates += 1

                        for agent in AGENTS:
                            _soft_update(target_actors[agent], actors[agent], args.tau)
                            _soft_update(target_critics[agent], critics[agent], args.tau)

            # Curriculum promotion by success window.
            stage_up_msg = None
            if (
                args.auto_curriculum
                and current_stage < max_stage
                and (not args.amp_enable or global_step >= args.amp_approach_steps)
            ):
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
                    stage_up_msg = (
                        f"curriculum: stage {prev_stage} -> {current_stage} "
                        f"(success_window={success_window:.2f} threshold={threshold:.2f})"
                    )

                    env.close()
                    env, raw_obs_batch, reset_infos = _build_env(args, stage=current_stage, seed=None)
                    obs_batch = _transform_obs_batch(
                        raw_obs_batch=raw_obs_batch,
                        infos_payload=reset_infos,
                        num_envs=args.num_envs,
                        adapter=obs_adapter,
                    )
                    if args.amp_enable:
                        amp_obs_current = _compute_amp_obs_batch(
                            raw_obs_batch=raw_obs_batch,
                            infos_payload=reset_infos,
                            num_envs=args.num_envs,
                            observation_mode=args.observation_mode,
                            done_mask=None,
                            use_reset_info_for_done=False,
                            joint_qpos_dim=amp_joint_qpos_dim,
                            joint_qvel_dim=amp_joint_qvel_dim,
                            include_root_height=bool(args.amp_include_root_height),
                        )
                    ep_returns[:] = 0.0
                    ep_lengths[:] = 0

            if global_step - last_print_step >= args.print_every_steps or global_step >= args.total_steps:
                elapsed = max(1e-6, time.time() - start_time)
                sps = int(global_step / elapsed)
                mean_return = float(np.mean(completed_returns[-20:])) if completed_returns else 0.0
                mean_len = float(np.mean(completed_lengths[-20:])) if completed_lengths else 0.0
                mean_success_20 = float(np.mean(completed_successes[-20:])) if completed_successes else 0.0
                success_window = float(
                    np.mean(completed_successes[-args.curriculum_window_episodes :])
                ) if completed_successes else 0.0

                c0 = float(np.mean(critic_loss_hist["h0"][-100:])) if critic_loss_hist["h0"] else 0.0
                c1 = float(np.mean(critic_loss_hist["h1"][-100:])) if critic_loss_hist["h1"] else 0.0
                a0 = float(np.mean(actor_loss_hist["h0"][-100:])) if actor_loss_hist["h0"] else 0.0
                a1 = float(np.mean(actor_loss_hist["h1"][-100:])) if actor_loss_hist["h1"] else 0.0
                amp_reward_mean = float(np.mean(amp_reward_hist[-2000:])) if amp_reward_hist else 0.0
                amp_weight_now = (
                    0.5 if global_step < args.amp_approach_steps else _amp_weight_for_stage(args, current_stage)
                )
                amp_disc_loss = float(np.mean(amp_disc_loss_hist[-100:])) if amp_disc_loss_hist else 0.0
                amp_d_real = float(np.mean(amp_d_real_hist[-100:])) if amp_d_real_hist else 0.0
                amp_d_fake = float(np.mean(amp_d_fake_hist[-100:])) if amp_d_fake_hist else 0.0

                msg = (
                    f"step={global_step}/{args.total_steps} sps={sps} stage={current_stage} "
                    f"ret20={mean_return:.2f} len20={mean_len:.1f} succ20={mean_success_20:.2f} "
                    f"succW={success_window:.2f} noise={current_noise:.3f} "
                    f"h0_closs={c0:.4f} h1_closs={c1:.4f} h0_aloss={a0:.4f} h1_aloss={a1:.4f}"
                )
                if args.amp_enable:
                    msg += (
                        f" amp_w={amp_weight_now:.2f} amp_r={amp_reward_mean:.3f} "
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
                if args.amp_enable:
                    algo_metrics["amp_weight"] = float(amp_weight_now)
                    algo_metrics["amp_reward_mean"] = float(amp_reward_mean)
                    algo_metrics["amp_disc_loss"] = float(amp_disc_loss)
                    algo_metrics["amp_d_real"] = float(amp_d_real)
                    algo_metrics["amp_d_fake"] = float(amp_d_fake)
                    algo_metrics["amp_policy_buffer_size"] = (
                        int(amp_policy_buffer.size) if amp_policy_buffer is not None else 0
                    )
                for agent in AGENTS:
                    if critic_loss_hist[agent]:
                        algo_metrics[f"{agent}/critic_loss"] = float(np.mean(critic_loss_hist[agent][-100:]))
                    if actor_loss_hist[agent]:
                        algo_metrics[f"{agent}/actor_loss"] = float(np.mean(actor_loss_hist[agent][-100:]))
                    if q_hist[agent]:
                        algo_metrics[f"{agent}/q_mean"] = float(np.mean(q_hist[agent][-100:]))
                task_metrics: Dict[str, float] = {
                    "stage": float(current_stage),
                }
                if args.task == "hug":
                    task_metrics["hug_contact_any_mean"] = _mean_recent(hug_contact_any_hist, n=2_000)
                    task_metrics["hug_dual_contact_mean"] = _mean_recent(hug_dual_contact_hist, n=2_000)
                    task_metrics["hug_hand_back_mean"] = _mean_recent(hug_hand_back_mean_hist, n=2_000)
                logger.log(
                    step=global_step,
                    common={
                        "sps": sps,
                        "ep_return_mean_20": mean_return,
                        "ep_len_mean_20": mean_len,
                        "ep_success_mean_20": mean_success_20,
                        "ep_success_window": success_window,
                        "curriculum_stage": current_stage,
                    },
                    algo=algo_metrics,
                    task=task_metrics,
                )

            if global_step - last_save_step >= args.save_every_steps or global_step >= args.total_steps:
                ckpt_path = os.path.join(args.save_dir, f"maddpg_step_{global_step:07d}.pt")
                payload = {
                    "algo": "maddpg",
                    "args": vars(args),
                    "global_step": global_step,
                    "stage": current_stage,
                    "collector_iter": collector_iter,
                    "grad_updates": grad_updates,
                    "actor_updates": actor_updates,
                    "episodes_in_stage": episodes_in_stage,
                    "last_print_step": last_print_step,
                    "last_save_step": last_save_step,
                    "observation_mode": args.observation_mode,
                    "uses_visual_encoder": obs_adapter.use_visual_encoder,
                    "uses_proprio_with_visual": obs_adapter.use_proprio_with_visual,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "n_agents": n_agents,
                    "actors": {agent: actors[agent].state_dict() for agent in AGENTS},
                    "critics": {agent: critics[agent].state_dict() for agent in AGENTS},
                    "target_actors": {agent: target_actors[agent].state_dict() for agent in AGENTS},
                    "target_critics": {agent: target_critics[agent].state_dict() for agent in AGENTS},
                    "actor_optim": {agent: actor_optim[agent].state_dict() for agent in AGENTS},
                    "critic_optim": {agent: critic_optim[agent].state_dict() for agent in AGENTS},
                }
                if args.amp_enable and amp_discriminator is not None:
                    payload["amp_discriminator"] = amp_discriminator.state_dict()
                    payload["amp_obs_dim"] = int(amp_obs_dim)
                    payload["amp_joint_qpos_dim"] = int(amp_joint_qpos_dim)
                    payload["amp_joint_qvel_dim"] = int(amp_joint_qvel_dim)
                    payload["amp_disc_optimizer"] = (
                        amp_disc_trainer.optimizer.state_dict() if amp_disc_trainer is not None else None
                    )
                torch.save(payload, ckpt_path)
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
