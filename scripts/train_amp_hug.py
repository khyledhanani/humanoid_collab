#!/usr/bin/env python3
"""Train hug task with AMP pre-trained policies.

This script continues training from AMP pre-trained checkpoints,
blending AMP style reward with hug task reward through curriculum.

Usage:
    python scripts/train_amp_hug.py --resume-from checkpoints/amp_pretrain/latest.pt

AMP weight schedule through curriculum:
    Stage 0: 40% AMP, 60% task
    Stage 1: 35% AMP, 65% task
    Stage 2: 30% AMP, 70% task
    Stage 3: 25% AMP, 75% task
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

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.normal import Normal
except ImportError as exc:
    raise ImportError(
        "Training requires PyTorch. Install with: pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.utils.exp_logging import ExperimentLogger
from humanoid_collab.amp.motion_data import load_motion_dataset
from humanoid_collab.amp.amp_obs import AMPObsBuilder
from humanoid_collab.amp.discriminator import AMPDiscriminator, AMPDiscriminatorTrainer
from humanoid_collab.amp.motion_buffer import MotionReplayBuffer
from humanoid_collab.amp.amp_reward import AMPRewardComputer


AGENTS = ("h0", "h1")

# AMP weight schedule per curriculum stage
AMP_WEIGHTS_BY_STAGE = {
    0: 0.40,
    1: 0.35,
    2: 0.30,
    3: 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMP + Hug task training")

    # Motion data
    parser.add_argument("--motion-data-dir", type=str, required=True)
    parser.add_argument("--motion-categories", type=str, nargs="+",
                        default=["standing", "walking", "reaching"])

    # Environment
    parser.add_argument("--physics-profile", type=str, default="default",
                        choices=available_physics_profiles())
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=30)
    parser.add_argument("--fixed-standing", action="store_true", default=False)
    parser.add_argument("--control-mode", type=str, default="all",
                        choices=["all", "arms_only"])

    # Training
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--anneal-lr", action="store_true")

    # Discriminator
    parser.add_argument("--disc-hidden-sizes", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--disc-batch-size", type=int, default=256)
    parser.add_argument("--n-disc-updates", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--freeze-discriminator", action="store_true",
                        help="Freeze discriminator (don't train during hug)")

    # AMP reward
    parser.add_argument("--amp-reward-scale", type=float, default=2.0)
    parser.add_argument("--clip-amp-reward", action="store_true", default=True)
    parser.add_argument("--amp-weight-override", type=float, default=None,
                        help="Override curriculum AMP weights with fixed value")
    parser.add_argument(
        "--approach-steps",
        type=int,
        default=500_000,
        help="Initial steps to train with 50/50 AMP+approach shaping before full hug reward",
    )

    # Curriculum
    parser.add_argument("--auto-curriculum", action="store_true")
    parser.add_argument("--curriculum-window-episodes", type=int, default=40)
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.6)
    parser.add_argument("--curriculum-min-updates-per-stage", type=int, default=25)

    # Network
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)

    # Checkpoint
    parser.add_argument("--resume-from", type=str, required=True,
                        help="Path to AMP pre-trained checkpoint")

    # Logging
    parser.add_argument("--log-dir", type=str, default="runs/amp_hug")
    parser.add_argument("--save-dir", type=str, default="checkpoints/amp_hug")
    parser.add_argument("--save-every-updates", type=int, default=25)
    parser.add_argument("--print-every-updates", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="humanoid-collab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="amp-hug")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")

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

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        v = self.v(x).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v

    def act(self, obs):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        log_prob = log_prob - correction
        return action, log_prob, v

    def evaluate_actions(self, obs, actions):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        clipped = torch.clamp(actions, -0.999999, 0.999999)
        pre_tanh = torch.atanh(clipped)
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        correction = torch.log(1.0 - clipped.pow(2) + 1e-6).sum(-1)
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
    amp_obs_t: np.ndarray
    amp_obs_t1: np.ndarray


def make_buffer(rollout_steps, obs_dim, act_dim, amp_obs_dim):
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


def compute_gae(rewards, values, dones, last_value, last_done, gamma, gae_lambda):
    advantages = np.zeros_like(rewards)
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


def get_amp_weight(stage: int, override: Optional[float] = None) -> float:
    """Get AMP reward weight for curriculum stage."""
    if override is not None:
        return override
    return AMP_WEIGHTS_BY_STAGE.get(stage, 0.3)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    device_str = str(device)

    print(f"Device: {device}")

    # Load motion data
    print(f"Loading motion data from {args.motion_data_dir}...")
    motion_dataset = load_motion_dataset(
        args.motion_data_dir,
        categories=tuple(args.motion_categories),
    )
    print(f"Loaded {motion_dataset.num_clips} clips")

    # Create environment
    env = HumanoidCollabEnv(
        task="hug",
        stage=args.stage,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        hold_target=args.hold_target,
        physics_profile=args.physics_profile,
        fixed_standing=args.fixed_standing,
        control_mode=args.control_mode,
    )

    max_stage = env.task_config.num_curriculum_stages - 1
    current_stage = int(max(0, min(args.stage, max_stage)))

    obs, _ = env.reset(seed=args.seed, options={"stage": current_stage})

    obs_dim = int(env.observation_space("h0").shape[0])
    act_dim = int(env.action_space("h0").shape[0])

    # Build AMP components
    amp_obs_builder = AMPObsBuilder(
        id_cache=env.id_cache,
        include_root_height=True,
        include_root_orientation=True,
        include_joint_positions=True,
        include_joint_velocities=True,
    )
    amp_obs_dim = amp_obs_builder.obs_dim

    motion_buffer = MotionReplayBuffer(motion_dataset, amp_obs_builder)

    discriminator = AMPDiscriminator(
        obs_dim=amp_obs_dim,
        hidden_sizes=tuple(args.disc_hidden_sizes),
    )

    disc_trainer = None
    if not args.freeze_discriminator:
        disc_trainer = AMPDiscriminatorTrainer(
            discriminator=discriminator,
            lr=args.disc_lr,
            lambda_gp=args.lambda_gp,
            n_updates=args.n_disc_updates,
            device=device_str,
        )

    amp_reward_computer = AMPRewardComputer(
        discriminator=discriminator,
        amp_obs_builder=amp_obs_builder,
        reward_scale=args.amp_reward_scale,
        clip_reward=args.clip_amp_reward,
        device=device_str,
    )

    # Create policies
    policies = {
        agent: ActorCritic(obs_dim, act_dim, args.hidden_size).to(device)
        for agent in AGENTS
    }
    optimizers = {
        agent: optim.Adam(policies[agent].parameters(), lr=args.lr, eps=1e-5)
        for agent in AGENTS
    }

    # Load checkpoint
    if not os.path.isfile(args.resume_from):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

    ckpt = torch.load(args.resume_from, map_location=device)
    for agent in AGENTS:
        policies[agent].load_state_dict(ckpt["policies"][agent])
    if "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
    if disc_trainer and "disc_optimizer" in ckpt:
        disc_trainer.optimizer.load_state_dict(ckpt["disc_optimizer"])

    print(f"Loaded checkpoint from {args.resume_from}")

    # Logging
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
    ep_return = 0.0
    ep_len = 0
    completed_returns = []
    completed_lengths = []
    completed_successes = []
    start_time = time.time()
    current_lr = float(args.lr)
    updates_in_stage = 0

    amp_reward_computer.reset(env.data, env.id_cache)

    print(f"\nStarting AMP+Hug training for {args.total_steps} steps...")
    print(
        f"Initial stage: {current_stage}, "
        f"approach_steps={args.approach_steps}, "
        f"curriculum AMP weight={get_amp_weight(current_stage, args.amp_weight_override):.2f}"
    )

    for update in range(1, updates + 1):
        # Phase schedule:
        # 1) optional approach phase: 50% AMP + 50% (distance + facing)
        # 2) curriculum phase: decaying AMP weight by stage
        in_approach_phase = global_step < args.approach_steps
        if in_approach_phase:
            amp_weight = 0.5
            task_weight = 0.5
        else:
            amp_weight = get_amp_weight(current_stage, args.amp_weight_override)
            task_weight = 1.0 - amp_weight

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / updates
            current_lr = args.lr * frac
        for opt in optimizers.values():
            for g in opt.param_groups:
                g["lr"] = current_lr

        buffers = {
            agent: make_buffer(args.rollout_steps, obs_dim, act_dim, amp_obs_dim)
            for agent in AGENTS
        }
        last_done = 0.0

        # Collect rollout
        for t in range(args.rollout_steps):
            amp_obs_t = amp_obs_builder.compute_obs_both(env.data)

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
                buffers[agent].amp_obs_t[t] = amp_obs_t[agent]

            next_obs, task_rewards, terminations, truncations, infos = env.step(action_dict)
            amp_obs_t1 = amp_obs_builder.compute_obs_both(env.data)
            amp_rewards = amp_reward_computer.compute_reward(env.data, env.id_cache)

            done = bool(terminations["h0"] or truncations["h0"])
            done_f = 1.0 if done else 0.0

            for agent in AGENTS:
                # Blend AMP and task rewards
                if in_approach_phase:
                    approach_task = (
                        float(infos[agent].get("r_distance", 0.0))
                        + float(infos[agent].get("r_facing", 0.0))
                    )
                    task_component = approach_task
                else:
                    task_component = task_rewards[agent]

                r_combined = amp_weight * amp_rewards[agent] + task_weight * task_component
                buffers[agent].rewards[t] = float(r_combined)
                buffers[agent].dones[t] = done_f
                buffers[agent].amp_obs_t1[t] = amp_obs_t1[agent]

            ep_return += float(buffers["h0"].rewards[t])
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
                obs, _ = env.reset(seed=None, options={"stage": current_stage})
                amp_reward_computer.reset(env.data, env.id_cache)
            else:
                obs = next_obs

        # GAE
        for agent in AGENTS:
            with torch.no_grad():
                last_obs_t = torch.as_tensor(obs[agent], dtype=torch.float32, device=device).unsqueeze(0)
                _, _, last_value_t = policies[agent].act(last_obs_t)
                last_value = float(last_value_t.item())

            adv, ret = compute_gae(
                buffers[agent].rewards, buffers[agent].values, buffers[agent].dones,
                last_value, last_done, args.gamma, args.gae_lambda
            )
            buffers[agent].advantages = adv
            buffers[agent].returns = ret

        # Train discriminator (if not frozen)
        disc_metrics = {"disc_loss": 0.0, "gp_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
        if disc_trainer is not None:
            real_obs_t, real_obs_t1 = motion_buffer.sample_torch(args.disc_batch_size, device=device_str)
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
            disc_metrics = disc_trainer.train_step(real_obs_t, real_obs_t1, fake_obs_t, fake_obs_t1)

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

            pg_losses, vf_losses, approx_kls = [], [], []

            idxs = np.arange(args.rollout_steps)
            for _ in range(args.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, args.rollout_steps, args.minibatch_size):
                    mb_idx = torch.as_tensor(idxs[start:start + args.minibatch_size], dtype=torch.long, device=device)

                    new_logp, entropy, new_values = policies[agent].evaluate_actions(
                        b_obs[mb_idx], b_actions[mb_idx]
                    )
                    log_ratio = new_logp - b_logp_old[mb_idx]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)

                    mb_adv = b_adv[mb_idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_losses.append(pg_loss.item())

                    vf_loss = 0.5 * ((new_values.view(-1) - b_returns[mb_idx]) ** 2).mean()
                    vf_losses.append(vf_loss.item())

                    loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy.mean()

                    optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policies[agent].parameters(), args.max_grad_norm)
                    optimizers[agent].step()

            metrics[f"{agent}/pg_loss"] = np.mean(pg_losses)
            metrics[f"{agent}/vf_loss"] = np.mean(vf_losses)
            metrics[f"{agent}/approx_kl"] = np.mean(approx_kls)

        # Curriculum advancement
        updates_in_stage += 1
        if args.auto_curriculum and (not in_approach_phase) and current_stage < max_stage:
            if len(completed_successes) >= args.curriculum_window_episodes:
                recent_success_rate = np.mean(completed_successes[-args.curriculum_window_episodes:])
                if (recent_success_rate >= args.curriculum_success_threshold and
                    updates_in_stage >= args.curriculum_min_updates_per_stage):
                    current_stage += 1
                    updates_in_stage = 0
                    print(f"\n>>> Advancing to stage {current_stage} (success rate: {recent_success_rate:.2f})")
                    obs, _ = env.reset(seed=None, options={"stage": current_stage})
                    amp_reward_computer.reset(env.data, env.id_cache)

        # Metrics
        metrics["amp/weight"] = amp_weight
        amp_style = amp_reward_computer.compute_reward_batch(
            np.concatenate([buffers["h0"].amp_obs_t, buffers["h1"].amp_obs_t], axis=0),
            np.concatenate([buffers["h0"].amp_obs_t1, buffers["h1"].amp_obs_t1], axis=0),
        )
        metrics["amp/reward_mean"] = float(np.mean(amp_style))
        metrics["amp/reward_std"] = float(np.std(amp_style))
        metrics["phase/approach"] = 1.0 if in_approach_phase else 0.0
        metrics["disc/loss"] = disc_metrics["disc_loss"]
        metrics["disc/d_real"] = disc_metrics["d_real"]
        metrics["disc/d_fake"] = disc_metrics["d_fake"]
        metrics["curriculum/stage"] = current_stage

        if completed_returns:
            metrics["episode/return_mean"] = float(np.mean(completed_returns[-100:]))
            metrics["episode/length_mean"] = float(np.mean(completed_lengths[-100:]))
        if completed_successes:
            metrics["episode/success_rate"] = float(np.mean(completed_successes[-args.curriculum_window_episodes:]))

        sps = global_step / max(time.time() - start_time, 1e-6)
        metrics["train/sps"] = sps
        metrics["train/global_step"] = global_step

        logger.log(metrics, step=global_step)

        if update % args.print_every_updates == 0:
            ret_str = f"{metrics.get('episode/return_mean', 0.0):.2f}" if completed_returns else "N/A"
            succ_str = f"{metrics.get('episode/success_rate', 0.0):.2f}" if completed_successes else "N/A"
            print(
                f"Update {update}/{updates} | Step {global_step} | "
                f"Stage {current_stage} | AMP {amp_weight:.2f} | "
                f"Phase {'approach' if in_approach_phase else 'hug'} | "
                f"Return {ret_str} | Success {succ_str}"
            )

        if update % args.save_every_updates == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{global_step}.pt")
            torch.save({
                "policies": {a: p.state_dict() for a, p in policies.items()},
                "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
                "discriminator": discriminator.state_dict(),
                "disc_optimizer": disc_trainer.optimizer.state_dict() if disc_trainer else None,
                "global_step": global_step,
                "stage": current_stage,
                "amp_weight": amp_weight,
            }, ckpt_path)

            torch.save({
                "policies": {a: p.state_dict() for a, p in policies.items()},
                "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
                "discriminator": discriminator.state_dict(),
                "disc_optimizer": disc_trainer.optimizer.state_dict() if disc_trainer else None,
                "global_step": global_step,
                "stage": current_stage,
                "amp_weight": amp_weight,
            }, os.path.join(args.save_dir, "latest.pt"))

    print(f"\nTraining complete!")
    logger.finish()
    env.close()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
