"""Render trained single-agent SAC walk-to-target checkpoint."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid_collab import HumanoidCollabEnv


HUMAN_RENDER_DELAY_S = 0.12


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
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

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mu, log_std

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return torch.tanh(mu)

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.sample()
        return torch.tanh(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SAC walk_to_target checkpoint in humanoid_collab."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sac_amp_walk_to_target",
        help="Directory used to resolve default checkpoint path as <checkpoint-dir>/latest.pt.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample from SAC policy distribution at eval time (default: deterministic mean action).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Optional stage override. Defaults to stage saved in checkpoint args.",
    )
    parser.add_argument(
        "--physics-profile",
        type=str,
        default=None,
        help="Optional physics profile override.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["all", "arms_only"],
        default=None,
        help="Optional control mode override.",
    )
    parser.add_argument(
        "--fixed-standing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional fixed-standing override. Use --no-fixed-standing for locomotion.",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        path = str(args.checkpoint)
    else:
        path = os.path.join(str(args.checkpoint_dir), "latest.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. Pass --checkpoint explicitly or set --checkpoint-dir."
        )
    return path


def make_env_from_ckpt_args(train_args: Dict[str, object], args: argparse.Namespace):
    backend = str(train_args.get("backend", "cpu"))
    if backend != "cpu":
        raise ValueError(
            f"Checkpoint backend='{backend}' is unsupported. "
            "Only CPU env backend is supported."
        )

    kwargs: Dict[str, object] = dict(
        task=str(train_args.get("task", "walk_to_target")),
        render_mode="human",
        stage=int(train_args.get("stage", 0)),
        horizon=int(train_args.get("horizon", 400)),
        frame_skip=int(train_args.get("frame_skip", 5)),
        hold_target=int(train_args.get("hold_target", 20)),
        physics_profile=str(train_args.get("physics_profile", "default")),
        fixed_standing=bool(train_args.get("fixed_standing", False)),
        control_mode=str(train_args.get("control_mode", "all")),
        observation_mode="proprio",
    )
    if args.stage is not None:
        kwargs["stage"] = int(args.stage)
    if args.physics_profile is not None:
        kwargs["physics_profile"] = str(args.physics_profile)
    if args.control_mode is not None:
        kwargs["control_mode"] = str(args.control_mode)
    if args.fixed_standing is not None:
        kwargs["fixed_standing"] = bool(args.fixed_standing)

    print(
        "render env config: "
        f"task={kwargs['task']} stage={kwargs['stage']} "
        f"physics_profile={kwargs['physics_profile']} "
        f"fixed_standing={kwargs['fixed_standing']} "
        f"control_mode={kwargs['control_mode']}"
    )
    return HumanoidCollabEnv(**kwargs)


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(args)
    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "actor" not in ckpt:
        raise KeyError(
            "Checkpoint format mismatch: expected key 'actor'. "
            "Use a checkpoint from scripts/train_sac_amp_walk_to_target.py."
        )

    train_args = ckpt.get("args", {})
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    hidden_size = int(train_args.get("hidden_size", 256))

    log_std_min = float(train_args.get("log_std_min", -5.0))
    log_std_max = float(train_args.get("log_std_max", 2.0))
    actor = GaussianActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    )
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = make_env_from_ckpt_args(train_args, args)
    env_obs_dim = int(env.observation_space("h0").shape[0])
    env_act_dim = int(env.action_space("h0").shape[0])
    if env_obs_dim != obs_dim:
        env.close()
        raise ValueError(
            f"Observation-dimension mismatch between checkpoint and env: "
            f"ckpt obs_dim={obs_dim}, env obs_dim={env_obs_dim}. "
            "Use matching env configuration for rendering."
        )
    if env_act_dim != act_dim:
        env.close()
        raise ValueError(
            f"Action-dimension mismatch between checkpoint and env: "
            f"ckpt act_dim={act_dim}, env act_dim={env_act_dim}. "
            "Use matching control settings for rendering."
        )

    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            obs, infos = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            ep_ret = 0.0
            steps = 0
            reason = "running"

            while not done:
                obs_t = torch.as_tensor(obs["h0"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if args.stochastic:
                        action_t = actor.sample_action(obs_t)
                    else:
                        action_t = actor.act_mean(obs_t)
                    action = action_t.squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                zero_h1 = np.zeros_like(action, dtype=np.float32)
                env_actions = {"h0": action, "h1": zero_h1}
                obs, rewards, terminations, truncations, infos = env.step(env_actions)
                env.render()
                time.sleep(HUMAN_RENDER_DELAY_S)

                ep_ret += float(rewards["h0"])
                steps += 1
                done = bool(terminations["h0"] or truncations["h0"])
                if done:
                    reason = str(infos["h0"].get("termination_reason", "unknown"))

            print(
                f"episode={ep + 1}/{args.episodes} return={ep_ret:.2f} "
                f"steps={steps} reason={reason}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
