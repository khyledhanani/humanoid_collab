#!/usr/bin/env python3
"""Pre-train humanoid policies with Adversarial Motion Priors (AMP).

This script trains policies to produce natural standing, walking, and reaching
motions by learning from motion capture reference data via a discriminator.

Usage:
    python scripts/train_amp_pretrain.py --motion-data-dir data/amass/processed

After pre-training, the policies can be transferred to the hug task with:
    python scripts/train_amp_hug.py --resume-from checkpoints/amp_pretrain/latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.normal import Normal
except ImportError as exc:
    raise ImportError(
        "AMP training requires PyTorch. Install training dependencies with: "
        "pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger
from humanoid_collab.amp.amp_config import AMPConfig
from humanoid_collab.amp.motion_data import load_motion_dataset
from humanoid_collab.amp.amp_obs import AMPObsBuilder
from humanoid_collab.amp.discriminator import AMPDiscriminator, AMPDiscriminatorTrainer
from humanoid_collab.amp.motion_buffer import MotionReplayBuffer
from humanoid_collab.amp.amp_reward import AMPRewardComputer


AGENTS = ("h0", "h1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMP pre-training for humanoid motion.")

    # Motion data
    parser.add_argument("--motion-data-dir", type=str, required=True,
                        help="Path to processed motion data directory")
    parser.add_argument("--motion-categories", type=str, nargs="+",
                        default=["standing", "walking", "reaching"],
                        help="Motion categories to use")

    # Environment
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--physics-profile", type=str, default="default",
                        choices=available_physics_profiles())
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=5)

    # Training
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true")

    # Discriminator
    parser.add_argument("--disc-hidden-sizes", type=int, nargs="+", default=[1024, 512])
    parser.add_argument(
        "--disc-objective",
        type=str,
        default="lsgan_gp",
        choices=["lsgan_gp", "wgan_gp"],
        help="Discriminator objective. lsgan_gp matches AMP paper reward shaping.",
    )
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--disc-batch-size", type=int, default=256)
    parser.add_argument("--n-disc-updates", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=10000,
                        help="Steps before starting discriminator training")

    # AMP reward
    parser.add_argument("--amp-reward-scale", type=float, default=2.0)
    parser.add_argument(
        "--clip-amp-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--alive-bonus", type=float, default=0.1)
    parser.add_argument("--energy-penalty", type=float, default=0.0005)
    parser.add_argument("--fall-penalty", type=float, default=10.0)

    # Network
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=4)

    # Logging
    parser.add_argument("--log-dir", type=str, default="runs/amp_pretrain")
    parser.add_argument("--save-dir", type=str, default="checkpoints/amp_pretrain")
    parser.add_argument("--save-every-updates", type=int, default=25)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--print-every-updates", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="amp-pretrain")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

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
    """Actor-Critic network for PPO."""

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
    # AMP-specific
    amp_obs_t: np.ndarray
    amp_obs_t1: np.ndarray


def make_buffer(rollout_steps: int, obs_dim: int, act_dim: int, amp_obs_dim: int) -> Buffer:
    return Buffer(
        obs=np.zeros((rollout_steps, obs_dim), dtype=np.float32),
        actions=np.zeros((rollout_steps, act_dim), dtype=np.float32),
        logp=np.zeros((rollout_steps,), dtype=np.float32),
        rewards=np.zeros((rollout_steps,), dtype=np.float32),
        dones=np.zeros((rollout_steps,), dtype=np.float32),
        values=np.zeros((rollout_steps,), dtype=np.float32),
        advantages=np.zeros((rollout_steps,), dtype=np.float32),
        returns=np.zeros((rollout_steps,), dtype=np.float32),
        amp_obs_t=np.zeros((rollout_steps, amp_obs_dim), dtype=np.float32),
        amp_obs_t1=np.zeros((rollout_steps, amp_obs_dim), dtype=np.float32),
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


def check_fallen(data, id_cache, height_threshold: float = 0.7) -> bool:
    """Check if either agent has fallen."""
    for agent in AGENTS:
        torso_pos = id_cache.get_torso_xpos(data, agent)
        if torso_pos[2] < height_threshold:
            return True
    return False


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    device_str = str(device)

    print(f"Device: {device}")
    print(f"Loading motion data from {args.motion_data_dir}...")

    # Load motion data
    motion_dataset = load_motion_dataset(
        args.motion_data_dir,
        categories=tuple(args.motion_categories),
    )
    print(f"Loaded {motion_dataset.num_clips} clips with {motion_dataset.total_transitions} transitions")

    # Create environment (no fixed standing, full control)
    env = HumanoidCollabEnv(
        task="hug",  # Use hug task but ignore task rewards
        stage=0,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        hold_target=30,
        physics_profile=args.physics_profile,
        fixed_standing=False,  # No fixed standing for AMP
        control_mode="all",    # Full body control
    )

    obs, _ = env.reset(seed=args.seed)

    obs_dim = int(env.observation_space("h0").shape[0])
    act_dim = int(env.action_space("h0").shape[0])

    # Build AMP observation builder
    amp_obs_builder = AMPObsBuilder(
        id_cache=env.id_cache,
        include_root_height=True,
        include_root_orientation=True,
        include_joint_positions=True,
        include_joint_velocities=True,
    )
    amp_obs_dim = amp_obs_builder.obs_dim
    print(f"AMP observation dim: {amp_obs_dim}")

    # Build motion replay buffer
    motion_buffer = MotionReplayBuffer(motion_dataset, amp_obs_builder)
    print(f"Motion buffer: {motion_buffer.num_transitions} transitions")

    # Create discriminator
    discriminator = AMPDiscriminator(
        obs_dim=amp_obs_dim,
        hidden_sizes=tuple(args.disc_hidden_sizes),
        activation="relu",
    )
    disc_trainer = AMPDiscriminatorTrainer(
        discriminator=discriminator,
        lr=args.disc_lr,
        lambda_gp=args.lambda_gp,
        n_updates=args.n_disc_updates,
        objective=args.disc_objective,
        device=device_str,
    )

    # Create AMP reward computer
    amp_reward_computer = AMPRewardComputer(
        discriminator=discriminator,
        amp_obs_builder=amp_obs_builder,
        reward_scale=args.amp_reward_scale,
        clip_reward=args.clip_amp_reward,
        device=device_str,
    )

    # Create policies
    policies: Dict[str, ActorCritic] = {
        agent: ActorCritic(obs_dim, act_dim, args.hidden_size).to(device)
        for agent in AGENTS
    }
    optimizers: Dict[str, optim.Optimizer] = {
        agent: optim.Adam(policies[agent].parameters(), lr=args.lr, eps=1e-5)
        for agent in AGENTS
    }

    # Logging setup
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

    # Training state
    global_step = 0
    updates = max(1, args.total_steps // args.rollout_steps)
    start_update = 1
    ep_return = 0.0
    ep_len = 0
    completed_returns = []
    completed_lengths = []
    start_time = time.time()
    current_lr = float(args.lr)

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

        ckpt = torch.load(args.resume_from, map_location=device)

        for agent in AGENTS:
            policies[agent].load_state_dict(ckpt["policies"][agent])
            if "optimizers" in ckpt and agent in ckpt["optimizers"]:
                optimizers[agent].load_state_dict(ckpt["optimizers"][agent])

        if "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
        if "disc_optimizer" in ckpt:
            disc_trainer.optimizer.load_state_dict(ckpt["disc_optimizer"])

        global_step = int(ckpt.get("global_step", 0))
        start_update = max(1, global_step // args.rollout_steps + 1)
        current_lr = float(ckpt.get("lr", args.lr))

        print(f"Resumed from {args.resume_from}: global_step={global_step}")

    # Initialize AMP reward computer
    amp_reward_computer.reset(env.data, env.id_cache)

    print(f"\nStarting AMP pre-training for {args.total_steps} steps...")

    for update in range(start_update, updates + 1):
        update_start = time.time()

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / updates
            current_lr = args.lr * frac
        for opt in optimizers.values():
            for g in opt.param_groups:
                g["lr"] = current_lr

        # Create buffers
        buffers = {
            agent: make_buffer(args.rollout_steps, obs_dim, act_dim, amp_obs_dim)
            for agent in AGENTS
        }
        last_done = 0.0

        # Collect rollout
        for t in range(args.rollout_steps):
            # Get AMP observation before step
            amp_obs_t = amp_obs_builder.compute_obs_both(env.data)

            # Sample actions
            action_dict = {}
            for agent in AGENTS:
                obs_t = torch.as_tensor(
                    obs[agent], dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action_t, logp_t, value_t = policies[agent].act(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
                action_dict[agent] = action_np

                buffers[agent].obs[t] = obs[agent]
                buffers[agent].actions[t] = action_np
                buffers[agent].logp[t] = float(logp_t.item())
                buffers[agent].values[t] = float(value_t.item())
                buffers[agent].amp_obs_t[t] = amp_obs_t[agent]

            # Step environment
            next_obs, _, terminations, truncations, infos = env.step(action_dict)

            # Get AMP observation after step
            amp_obs_t1 = amp_obs_builder.compute_obs_both(env.data)

            # Check if fallen
            fallen = check_fallen(env.data, env.id_cache)

            # Compute AMP reward
            amp_rewards = amp_reward_computer.compute_reward(env.data, env.id_cache)

            # Compute total pre-training reward
            ctrl = np.concatenate([action_dict[a] for a in AGENTS])
            ctrl_penalty = args.energy_penalty * np.sum(ctrl ** 2)

            done = bool(terminations["h0"] or truncations["h0"] or fallen)
            done_f = 1.0 if done else 0.0

            for agent in AGENTS:
                # Pre-training reward: AMP + alive bonus - energy penalty - fall penalty
                r = amp_rewards[agent]
                r += args.alive_bonus
                r -= ctrl_penalty / 2  # Split between agents
                if fallen:
                    r -= args.fall_penalty

                buffers[agent].rewards[t] = float(r)
                buffers[agent].dones[t] = done_f
                buffers[agent].amp_obs_t1[t] = amp_obs_t1[agent]

            ep_return += float(buffers["h0"].rewards[t])
            ep_len += 1
            global_step += 1
            last_done = done_f

            if done:
                completed_returns.append(ep_return)
                completed_lengths.append(ep_len)
                ep_return = 0.0
                ep_len = 0
                obs, _ = env.reset(seed=None)
                amp_reward_computer.reset(env.data, env.id_cache)
            else:
                obs = next_obs

        # Compute GAE
        for agent in AGENTS:
            with torch.no_grad():
                last_obs_t = torch.as_tensor(
                    obs[agent], dtype=torch.float32, device=device
                ).unsqueeze(0)
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

        # Train discriminator (after warmup)
        disc_metrics = {"disc_loss": 0.0, "gp_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
        if global_step > args.warmup_steps:
            # Sample real transitions from motion buffer
            real_obs_t, real_obs_t1 = motion_buffer.sample_torch(
                args.disc_batch_size, device=device_str
            )

            # Sample fake transitions from both agents for agent-agnostic style.
            fake_obs_t_all = np.concatenate(
                [buffers["h0"].amp_obs_t, buffers["h1"].amp_obs_t], axis=0
            )
            fake_obs_t1_all = np.concatenate(
                [buffers["h0"].amp_obs_t1, buffers["h1"].amp_obs_t1], axis=0
            )
            fake_count = min(args.disc_batch_size, fake_obs_t_all.shape[0])
            fake_idx = np.random.choice(fake_obs_t_all.shape[0], size=fake_count, replace=False)
            fake_obs_t = torch.as_tensor(fake_obs_t_all[fake_idx], device=device)
            fake_obs_t1 = torch.as_tensor(fake_obs_t1_all[fake_idx], device=device)

            disc_metrics = disc_trainer.train_step(
                real_obs_t, real_obs_t1,
                fake_obs_t, fake_obs_t1,
            )

        # PPO update
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
                    mb_idx = idxs[start:start + args.minibatch_size]
                    mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

                    new_logp, entropy, new_values = policies[agent].evaluate_actions(
                        b_obs[mb_idx_t], b_actions[mb_idx_t]
                    )
                    log_ratio = new_logp - b_logp_old[mb_idx_t]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)
                        clip_frac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        clip_fracs.append(clip_frac)

                    mb_adv = b_adv[mb_idx_t]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_losses.append(pg_loss.item())

                    new_values = new_values.view(-1)
                    vf_loss = 0.5 * ((new_values - b_returns[mb_idx_t]) ** 2).mean()
                    vf_losses.append(vf_loss.item())

                    ent_loss = -entropy.mean()
                    ent_losses.append(ent_loss.item())

                    loss = pg_loss + args.vf_coef * vf_loss + args.ent_coef * ent_loss

                    optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policies[agent].parameters(), args.max_grad_norm)
                    optimizers[agent].step()

            metrics[f"{agent}/pg_loss"] = np.mean(pg_losses)
            metrics[f"{agent}/vf_loss"] = np.mean(vf_losses)
            metrics[f"{agent}/entropy"] = -np.mean(ent_losses)
            metrics[f"{agent}/approx_kl"] = np.mean(approx_kls)
            metrics[f"{agent}/clip_frac"] = np.mean(clip_fracs)

        # Compute AMP reward stats
        # Report pure AMP style reward (without alive/energy/fall shaping).
        amp_style = amp_reward_computer.compute_reward_batch(
            np.concatenate([buffers["h0"].amp_obs_t, buffers["h1"].amp_obs_t], axis=0),
            np.concatenate([buffers["h0"].amp_obs_t1, buffers["h1"].amp_obs_t1], axis=0),
        )
        metrics["amp/reward_mean"] = float(np.mean(amp_style))
        metrics["amp/reward_std"] = float(np.std(amp_style))

        # Discriminator metrics
        metrics["disc/loss"] = disc_metrics["disc_loss"]
        metrics["disc/gp_loss"] = disc_metrics["gp_loss"]
        metrics["disc/d_real"] = disc_metrics["d_real"]
        metrics["disc/d_fake"] = disc_metrics["d_fake"]

        # Episode stats
        if completed_returns:
            metrics["episode/return_mean"] = float(np.mean(completed_returns[-100:]))
            metrics["episode/length_mean"] = float(np.mean(completed_lengths[-100:]))

        # Training stats
        update_time = time.time() - update_start
        total_time = time.time() - start_time
        sps = global_step / max(total_time, 1e-6)
        metrics["train/sps"] = sps
        metrics["train/lr"] = current_lr
        metrics["train/global_step"] = global_step

        # Log metrics (ExperimentLogger expects structured namespaces).
        common = {
            "sps": sps,
            "global_step": global_step,
        }
        if "episode/return_mean" in metrics:
            common["ep_return_mean_100"] = metrics["episode/return_mean"]
        if "episode/length_mean" in metrics:
            common["ep_len_mean_100"] = metrics["episode/length_mean"]

        algo = {
            "lr": current_lr,
            "amp_reward_mean": metrics["amp/reward_mean"],
            "amp_reward_std": metrics["amp/reward_std"],
            "disc_loss": metrics["disc/loss"],
            "disc_gp_loss": metrics["disc/gp_loss"],
            "disc_d_real": metrics["disc/d_real"],
            "disc_d_fake": metrics["disc/d_fake"],
        }
        for agent in AGENTS:
            for key in ("pg_loss", "vf_loss", "entropy", "approx_kl", "clip_frac"):
                k = f"{agent}/{key}"
                if k in metrics:
                    algo[k] = metrics[k]

        logger.log(step=global_step, common=common, algo=algo)

        # Print progress
        if update % args.print_every_updates == 0:
            ret_str = f"{metrics.get('episode/return_mean', 0.0):.2f}" if completed_returns else "N/A"
            print(
                f"Update {update}/{updates} | "
                f"Step {global_step} | "
                f"SPS {sps:.0f} | "
                f"Return {ret_str} | "
                f"AMP {metrics['amp/reward_mean']:.3f} | "
                f"D_real {disc_metrics['d_real']:.3f} D_fake {disc_metrics['d_fake']:.3f}"
            )

        # Save checkpoint
        if update % args.save_every_updates == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{global_step}.pt")
            torch.save({
                "policies": {a: p.state_dict() for a, p in policies.items()},
                "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
                "discriminator": discriminator.state_dict(),
                "disc_optimizer": disc_trainer.optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "lr": current_lr,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "amp_obs_dim": amp_obs_dim,
            }, ckpt_path)

            # Also save as "latest"
            latest_path = os.path.join(args.save_dir, "latest.pt")
            torch.save({
                "policies": {a: p.state_dict() for a, p in policies.items()},
                "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
                "discriminator": discriminator.state_dict(),
                "disc_optimizer": disc_trainer.optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "lr": current_lr,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "amp_obs_dim": amp_obs_dim,
            }, latest_path)

    # Final save
    final_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "policies": {a: p.state_dict() for a, p in policies.items()},
        "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
        "discriminator": discriminator.state_dict(),
        "disc_optimizer": disc_trainer.optimizer.state_dict(),
        "global_step": global_step,
        "update": updates,
        "lr": current_lr,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "amp_obs_dim": amp_obs_dim,
    }, final_path)

    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    logger.finish()
    env.close()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
