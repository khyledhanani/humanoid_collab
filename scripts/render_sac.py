"""Render trained SAC checkpoint for staged hug (train_sac.py)."""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.partners.scripted import ScriptedPartner


HUMAN_RENDER_DELAY_S = 0.12


class SACGaussianActor(nn.Module):
    """SAC actor network matching scripts/train_sac.py."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Linear(hidden_size, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        action, _ = self.forward(obs)
        return action.cpu().numpy()

    @torch.no_grad()
    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        h = self.trunk(obs)
        return torch.tanh(self.mu_head(h)).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SAC staged-hug checkpoint in humanoid_collab."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sac_hug_stage_ab",
        help="Directory used to resolve default checkpoint path as <checkpoint-dir>/final.pt.",
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
        "--task",
        type=str,
        default="hug",
        help="Task name (default: hug).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Optional stage override. Defaults to stage saved in checkpoint.",
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=30)
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="train_fast",
        choices=available_physics_profiles(),
    )
    parser.add_argument(
        "--partner-mode",
        type=str,
        default="zero",
        choices=["zero", "scripted"],
        help="zero=Stage A behavior, scripted=Stage B behavior.",
    )
    parser.add_argument(
        "--h1-weld-mode",
        type=str,
        default="torso",
        choices=["torso", "lower_body"],
        help="torso=Stage A setup, lower_body=Stage B setup.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override actor hidden size. Defaults to value inferred from checkpoint actor weights.",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        path = str(args.checkpoint)
    else:
        path = os.path.join(str(args.checkpoint_dir), "final.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. Pass --checkpoint explicitly or set --checkpoint-dir."
        )
    return path


def _infer_actor_dims(actor_state: dict) -> Tuple[int, int, int]:
    if "trunk.0.weight" not in actor_state or "mu_head.weight" not in actor_state:
        raise KeyError(
            "Checkpoint actor format mismatch: expected keys 'trunk.0.weight' and 'mu_head.weight'."
        )
    obs_dim = int(actor_state["trunk.0.weight"].shape[1])
    hidden_size = int(actor_state["trunk.0.weight"].shape[0])
    act_dim = int(actor_state["mu_head.weight"].shape[0])
    return obs_dim, act_dim, hidden_size


def _build_arm_actuator_positions(env: HumanoidCollabEnv, partner: str = "h1") -> list[int]:
    arm_tokens = ("shoulder", "elbow")
    positions: list[int] = []
    for pos, full_idx in enumerate(env._control_actuator_idx[partner]):
        act_name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(full_idx)
        )
        if act_name and any(t in act_name for t in arm_tokens):
            positions.append(pos)
    return positions


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(args)
    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "actor" not in ckpt:
        raise KeyError(
            "Checkpoint format mismatch: expected key 'actor'. "
            "Use a checkpoint produced by scripts/train_sac.py."
        )

    actor_state = ckpt["actor"]
    obs_dim_ckpt, act_dim_ckpt, inferred_hidden = _infer_actor_dims(actor_state)
    hidden_size = int(args.hidden_size) if args.hidden_size is not None else inferred_hidden
    stage = int(ckpt.get("stage", 0)) if args.stage is None else int(args.stage)

    actor = SACGaussianActor(
        obs_dim=obs_dim_ckpt,
        act_dim=act_dim_ckpt,
        hidden_size=hidden_size,
    )
    actor.load_state_dict(actor_state)
    actor.eval()

    env_kwargs = dict(
        task=args.task,
        render_mode="human",
        stage=stage,
        horizon=int(args.horizon),
        frame_skip=int(args.frame_skip),
        hold_target=int(args.hold_target),
        physics_profile=args.physics_profile,
        fixed_standing=False,
        control_mode="all",
        weld_h1_only=True,
        h1_weld_mode=args.h1_weld_mode,
    )
    print(
        "render env config: "
        f"task={env_kwargs['task']} stage={env_kwargs['stage']} "
        f"physics_profile={env_kwargs['physics_profile']} "
        f"partner_mode={args.partner_mode} h1_weld_mode={env_kwargs['h1_weld_mode']}"
    )

    env = HumanoidCollabEnv(**env_kwargs)
    env_obs_dim = int(env.observation_space("h0").shape[0])
    env_act_dim = int(env.action_space("h0").shape[0])
    if env_obs_dim != obs_dim_ckpt:
        env.close()
        raise ValueError(
            f"Observation-dimension mismatch between checkpoint and env: "
            f"ckpt obs_dim={obs_dim_ckpt}, env obs_dim={env_obs_dim}."
        )
    if env_act_dim != act_dim_ckpt:
        env.close()
        raise ValueError(
            f"Action-dimension mismatch between checkpoint and env: "
            f"ckpt act_dim={act_dim_ckpt}, env act_dim={env_act_dim}."
        )

    scripted_partner = None
    if args.partner_mode == "scripted":
        arm_positions = _build_arm_actuator_positions(env, partner="h1")
        scripted_partner = ScriptedPartner(
            id_cache=env.id_cache,
            arm_actuator_positions=arm_positions,
            act_dim=env_act_dim,
        )
        print(f"scripted partner arm positions: {arm_positions}")

    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            obs, infos = env.reset(seed=int(rng.integers(0, 2**31 - 1)), options={"stage": stage})
            if scripted_partner is not None:
                scripted_partner.reset()

            done = False
            ep_ret = 0.0
            steps = 0
            reason = "running"

            while not done:
                obs_t = torch.as_tensor(obs["h0"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if args.stochastic:
                        action_h0 = actor.get_action(obs_t)[0]
                    else:
                        action_h0 = actor.get_deterministic_action(obs_t)[0]
                action_h0 = np.clip(action_h0, -1.0, 1.0).astype(np.float32)

                if scripted_partner is not None:
                    action_h1 = scripted_partner.get_action(env.data)
                else:
                    action_h1 = np.zeros(env_act_dim, dtype=np.float32)

                obs, rewards, terminations, truncations, infos = env.step(
                    {"h0": action_h0, "h1": action_h1}
                )
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
