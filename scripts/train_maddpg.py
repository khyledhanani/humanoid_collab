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
from typing import Dict, List, Optional, Tuple

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

from humanoid_collab import HumanoidCollabEnv, SubprocHumanoidCollabVecEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger


AGENTS = ("h0", "h1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MADDPG on HumanoidCollabEnv.")

    parser.add_argument("--task", type=str, default="handshake", choices=["hug", "handshake", "box_lift"])
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--physics-profile", type=str, default="default", choices=available_physics_profiles())
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=30)
    parser.add_argument("--fixed-standing", action="store_true", default=True)
    parser.add_argument("--control-mode", type=str, default="arms_only", choices=["all", "arms_only"])
    parser.add_argument("--observation-mode", type=str, default="proprio", choices=["proprio"])

    parser.add_argument("--total-steps", type=int, default=800_000, help="Total environment transitions.")
    parser.add_argument("--num-envs", type=int, default=1)
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
    parser.add_argument("--print-every-steps", type=int, default=2_048)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="maddpg")
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
    ):
        if capacity <= 0:
            raise ValueError("Replay capacity must be > 0")
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.num_agents, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
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
        for i in range(batch_n):
            idx = self.ptr
            self.obs[idx] = obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_obs[idx] = next_obs[i]
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


def _make_env_kwargs(args: argparse.Namespace, stage: int) -> Dict[str, object]:
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
    )


def _format_reset_obs(obs: Dict[str, np.ndarray], num_envs: int) -> Dict[str, np.ndarray]:
    if num_envs == 1:
        return {
            agent: np.asarray(obs[agent], dtype=np.float32)[None, ...]
            for agent in AGENTS
        }
    return {
        agent: np.asarray(obs[agent], dtype=np.float32)
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
    out = {agent: np.asarray(next_obs_batch[agent], dtype=np.float32).copy() for agent in AGENTS}
    for env_idx, done in enumerate(done_mask):
        if not bool(done):
            continue
        info_env = infos[env_idx]
        for agent in AGENTS:
            final_obs = info_env[agent].get("final_observation")
            if final_obs is not None:
                out[agent][env_idx] = np.asarray(final_obs, dtype=np.float32)
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
        and max_stage >= 1
    ):
        current_stage = 1
        print(
            "curriculum guard: using stage 1 instead of stage 0 for "
            "handshake + fixed-standing + arms_only."
        )

    env, obs_batch, _ = _build_env(args, stage=current_stage, seed=args.seed)

    if obs_batch["h0"].ndim != 2:
        env.close()
        raise ValueError(
            "MADDPG currently supports vector observations only. "
            "Use --observation-mode proprio."
        )

    obs_dim = int(obs_batch["h0"].shape[-1])
    act_dim = int(env.action_space("h0").shape[0])
    n_agents = len(AGENTS)
    joint_input_dim = n_agents * (obs_dim + act_dim)

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
                next_obs_for_buffer = {
                    agent: np.asarray(next_obs_dict[agent], dtype=np.float32)[None, ...]
                    for agent in AGENTS
                }
                next_obs_for_rollout = next_obs_for_buffer

                rewards_batch = {
                    agent: np.asarray([float(rewards_dict[agent])], dtype=np.float32)
                    for agent in AGENTS
                }
                infos_list = [infos]

                if bool(done_mask[0]):
                    reset_obs, _ = env.reset(seed=None, options={"stage": int(current_stage)})
                    next_obs_for_rollout = {
                        agent: np.asarray(reset_obs[agent], dtype=np.float32)[None, ...]
                        for agent in AGENTS
                    }

            else:
                next_obs_vec, rewards_vec, terminations, truncations, infos_list = env.step(actions_batch)
                done_mask, reasons = _extract_done_and_reasons_vec(terminations, truncations, infos_list)

                next_obs_for_rollout = {
                    agent: np.asarray(next_obs_vec[agent], dtype=np.float32)
                    for agent in AGENTS
                }
                next_obs_for_buffer = _maybe_replace_final_obs_from_infos(
                    next_obs_batch=next_obs_for_rollout,
                    done_mask=done_mask,
                    infos=infos_list,
                )
                rewards_batch = {
                    agent: np.asarray(rewards_vec[agent], dtype=np.float32)
                    for agent in AGENTS
                }

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
                    stage_up_msg = (
                        f"curriculum: stage {prev_stage} -> {current_stage} "
                        f"(success_window={success_window:.2f} threshold={threshold:.2f})"
                    )

                    env.close()
                    env, obs_batch, _ = _build_env(args, stage=current_stage, seed=None)
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

                print(
                    f"step={global_step}/{args.total_steps} sps={sps} stage={current_stage} "
                    f"ret20={mean_return:.2f} len20={mean_len:.1f} succ20={mean_success_20:.2f} "
                    f"succW={success_window:.2f} noise={current_noise:.3f} "
                    f"h0_closs={c0:.4f} h1_closs={c1:.4f} h0_aloss={a0:.4f} h1_aloss={a1:.4f}"
                )
                if stage_up_msg is not None:
                    print(stage_up_msg)
                last_print_step = global_step

                algo_metrics = {
                    "replay_size": replay.size,
                    "noise_std": current_noise,
                    "grad_updates": grad_updates,
                    "actor_updates": actor_updates,
                }
                for agent in AGENTS:
                    if critic_loss_hist[agent]:
                        algo_metrics[f"{agent}/critic_loss"] = float(np.mean(critic_loss_hist[agent][-100:]))
                    if actor_loss_hist[agent]:
                        algo_metrics[f"{agent}/actor_loss"] = float(np.mean(actor_loss_hist[agent][-100:]))
                    if q_hist[agent]:
                        algo_metrics[f"{agent}/q_mean"] = float(np.mean(q_hist[agent][-100:]))
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
                    task={
                        "stage": current_stage,
                    },
                )

            if global_step - last_save_step >= args.save_every_steps or global_step >= args.total_steps:
                ckpt_path = os.path.join(args.save_dir, f"maddpg_step_{global_step:07d}.pt")
                payload = {
                    "algo": "maddpg",
                    "args": vars(args),
                    "global_step": global_step,
                    "stage": current_stage,
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
