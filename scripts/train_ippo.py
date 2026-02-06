"""Train IPPO policies for HumanoidCollabEnv.

Default settings target fixed-standing, arms-only handshake training.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

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

try:
    from humanoid_collab import MJXHumanoidCollabEnv
except Exception:
    MJXHumanoidCollabEnv = None  # type: ignore[assignment]

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore[assignment]


AGENTS = ("h0", "h1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IPPO on HumanoidCollabEnv.")

    parser.add_argument("--task", type=str, default="handshake", choices=["hug", "handshake", "box_lift"])
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "mjx"])
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
    parser.add_argument("--print-every-updates", type=int, default=1)
    parser.add_argument("--no-tensorboard", action="store_true")

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
    if args.backend == "mjx":
        if MJXHumanoidCollabEnv is None:
            raise RuntimeError("MJX backend unavailable. Install with: pip install -e '.[mjx]'")
        return MJXHumanoidCollabEnv(**kwargs)
    return HumanoidCollabEnv(**kwargs)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)

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

    use_tb = (not args.no_tensorboard) and SummaryWriter is not None
    writer = SummaryWriter(args.log_dir) if use_tb else None
    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    updates = max(1, args.total_steps // args.rollout_steps)
    ep_return = 0.0
    ep_len = 0
    completed_returns = []
    completed_lengths = []
    start_time = time.time()

    for update in range(1, updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / updates
            lr_now = args.lr * frac
            for opt in optimizers.values():
                for g in opt.param_groups:
                    g["lr"] = lr_now

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
                completed_returns.append(ep_return)
                completed_lengths.append(ep_len)
                ep_return = 0.0
                ep_len = 0
                obs, _ = env.reset(seed=None)
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
            b_values_old = torch.as_tensor(b.values, dtype=torch.float32, device=device)

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
                        clip_frac = (
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                        )

                    mb_adv = b_adv[mb_idx_t]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
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

        if update % args.print_every_updates == 0:
            print(
                f"update={update}/{updates} step={global_step} sps={sps} "
                f"ret20={mean_return:.2f} len20={mean_ep_len:.1f} "
                f"h0_kl={metrics['h0']['approx_kl']:.4f} h1_kl={metrics['h1']['approx_kl']:.4f}"
            )

        if writer is not None:
            writer.add_scalar("charts/sps", sps, global_step)
            writer.add_scalar("charts/ep_return_mean_20", mean_return, global_step)
            writer.add_scalar("charts/ep_len_mean_20", mean_ep_len, global_step)
            for agent in AGENTS:
                for k, v in metrics[agent].items():
                    writer.add_scalar(f"{agent}/{k}", v, global_step)

        if update % args.save_every_updates == 0 or update == updates:
            ckpt_path = os.path.join(args.save_dir, f"ippo_update_{update:06d}.pt")
            payload = {
                "args": vars(args),
                "global_step": global_step,
                "update": update,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "policies": {a: policies[a].state_dict() for a in AGENTS},
                "optimizers": {a: optimizers[a].state_dict() for a in AGENTS},
            }
            torch.save(payload, ckpt_path)

    env.close()
    if writer is not None:
        writer.close()
    print("Training complete.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
