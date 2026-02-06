"""Render trained IPPO policies in the humanoid collaboration environment."""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from humanoid_collab import HumanoidCollabEnv


AGENTS = ("h0", "h1")


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

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std).expand_as(mu)
        v = self.v(x).squeeze(-1)
        return mu, std, v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render IPPO checkpoint in humanoid_collab.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true", help="Use tanh(mu) instead of sampling.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_env_from_ckpt_args(train_args: Dict[str, object]):
    backend = str(train_args.get("backend", "cpu"))
    if backend != "cpu":
        raise ValueError(
            f"Checkpoint backend='{backend}' is unsupported. "
            "Only CPU checkpoints are supported."
        )
    kwargs = dict(
        task=str(train_args.get("task", "handshake")),
        render_mode="human",
        stage=int(train_args.get("stage", 0)),
        horizon=int(train_args.get("horizon", 1000)),
        frame_skip=int(train_args.get("frame_skip", 5)),
        hold_target=int(train_args.get("hold_target", 30)),
        physics_profile=str(train_args.get("physics_profile", "default")),
        fixed_standing=bool(train_args.get("fixed_standing", False)),
        control_mode=str(train_args.get("control_mode", "all")),
    )
    return HumanoidCollabEnv(**kwargs)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    hidden_size = int(train_args.get("hidden_size", 256))

    policies = {a: ActorCritic(obs_dim, act_dim, hidden_size) for a in AGENTS}
    for agent in AGENTS:
        policies[agent].load_state_dict(ckpt["policies"][agent])
        policies[agent].eval()

    env = make_env_from_ckpt_args(train_args)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            obs, infos = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            ep_ret = 0.0
            steps = 0
            reason = "running"
            while not done:
                actions = {}
                for agent in AGENTS:
                    obs_t = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        mu, std, _ = policies[agent].forward(obs_t)
                    if args.deterministic:
                        action = torch.tanh(mu)
                    else:
                        action = torch.tanh(mu + std * torch.randn_like(std))
                    actions[agent] = action.squeeze(0).cpu().numpy().astype(np.float32)

                obs, rewards, terminations, truncations, infos = env.step(actions)
                env.render()
                ep_ret += float(rewards["h0"])
                steps += 1
                done = bool(terminations["h0"] or truncations["h0"])
                if done:
                    reason = str(infos["h0"].get("termination_reason", "unknown"))

            print(f"episode={ep + 1}/{args.episodes} return={ep_ret:.2f} steps={steps} reason={reason}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
