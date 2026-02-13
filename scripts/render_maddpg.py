"""Render trained MADDPG actors in HumanoidCollabEnv."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.vision import load_frozen_encoder_from_vae_checkpoint


AGENTS = ("h0", "h1")
HUMAN_RENDER_DELAY_S = 0.12


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MADDPG checkpoint in humanoid_collab.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0, help="Optional exploration noise at eval time.")
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
        "--encoder-checkpoint",
        type=str,
        default=None,
        help="Optional VAE checkpoint override for visual observations.",
    )
    parser.add_argument(
        "--vision-use-proprio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override proprio+latent concatenation for visual observations.",
    )
    parser.add_argument(
        "--fixed-standing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fixed-standing behavior. Use --no-fixed-standing for locomotion.",
    )
    return parser.parse_args()


def make_env_from_ckpt_args(train_args: Dict[str, object], args: argparse.Namespace):
    backend = str(train_args.get("backend", "cpu"))
    if backend != "cpu":
        raise ValueError(
            f"Checkpoint backend='{backend}' is unsupported. "
            "Only CPU env backend is supported."
        )

    kwargs: Dict[str, object] = dict(
        task=str(train_args.get("task", "handshake")),
        render_mode="human",
        stage=int(train_args.get("stage", 1)),
        horizon=int(train_args.get("horizon", 600)),
        frame_skip=int(train_args.get("frame_skip", 5)),
        hold_target=int(train_args.get("hold_target", 30)),
        physics_profile=str(train_args.get("physics_profile", "default")),
        fixed_standing=bool(train_args.get("fixed_standing", False)),
        control_mode=str(train_args.get("control_mode", "all")),
        observation_mode=str(train_args.get("observation_mode", "proprio")),
        obs_rgb_width=int(train_args.get("obs_rgb_width", 84)),
        obs_rgb_height=int(train_args.get("obs_rgb_height", 84)),
    )
    if args.stage is not None:
        kwargs["stage"] = int(args.stage)
    if args.physics_profile is not None:
        kwargs["physics_profile"] = str(args.physics_profile)
    if args.control_mode is not None:
        kwargs["control_mode"] = str(args.control_mode)
    if args.fixed_standing is not None:
        kwargs["fixed_standing"] = bool(args.fixed_standing)
    vision_use_proprio = bool(train_args.get("vision_use_proprio", True))
    if args.vision_use_proprio is not None:
        vision_use_proprio = bool(args.vision_use_proprio)
    kwargs["emit_proprio_info"] = bool(kwargs["observation_mode"] in {"rgb", "gray"} and vision_use_proprio)

    print(
        "render env config: "
        f"task={kwargs['task']} stage={kwargs['stage']} "
        f"physics_profile={kwargs['physics_profile']} "
        f"fixed_standing={kwargs['fixed_standing']} "
        f"control_mode={kwargs['control_mode']} "
        f"observation_mode={kwargs['observation_mode']} "
        f"emit_proprio_info={kwargs['emit_proprio_info']}"
    )
    return HumanoidCollabEnv(**kwargs)


@dataclass
class RenderObsAdapter:
    use_visual_encoder: bool
    use_proprio_with_visual: bool
    encoder_bundle: Any
    device: torch.device


def _build_obs_adapter(
    train_args: Dict[str, object],
    cli_args: argparse.Namespace,
    device: torch.device,
) -> RenderObsAdapter:
    observation_mode = str(train_args.get("observation_mode", "proprio"))
    if observation_mode == "proprio":
        return RenderObsAdapter(
            use_visual_encoder=False,
            use_proprio_with_visual=False,
            encoder_bundle=None,
            device=device,
        )
    encoder_ckpt = cli_args.encoder_checkpoint
    if encoder_ckpt is None:
        encoder_ckpt = train_args.get("encoder_checkpoint")
    if encoder_ckpt is None:
        raise ValueError(
            "Visual checkpoint render requires an encoder checkpoint. "
            "Pass --encoder-checkpoint or include encoder_checkpoint in training args."
        )
    bundle = load_frozen_encoder_from_vae_checkpoint(str(encoder_ckpt), device=device)
    if bundle.observation_mode != observation_mode:
        raise ValueError(
            f"Encoder observation_mode='{bundle.observation_mode}' does not match "
            f"checkpoint observation_mode='{observation_mode}'."
        )
    use_prop = bool(train_args.get("vision_use_proprio", True))
    if cli_args.vision_use_proprio is not None:
        use_prop = bool(cli_args.vision_use_proprio)
    return RenderObsAdapter(
        use_visual_encoder=True,
        use_proprio_with_visual=use_prop,
        encoder_bundle=bundle,
        device=device,
    )


def _transform_obs(
    obs: Dict[str, np.ndarray],
    infos: Dict[str, Dict[str, object]],
    adapter: RenderObsAdapter,
) -> Dict[str, np.ndarray]:
    if not adapter.use_visual_encoder:
        return {agent: np.asarray(obs[agent], dtype=np.float32) for agent in AGENTS}
    out: Dict[str, np.ndarray] = {}
    for agent in AGENTS:
        frame = np.asarray(obs[agent])
        x = torch.as_tensor(frame[None, ...], dtype=torch.float32, device=adapter.device) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            z = adapter.encoder_bundle.vae.encode_mean(x).squeeze(0).cpu().numpy().astype(np.float32)
        if adapter.use_proprio_with_visual:
            p = infos[agent].get("proprio_obs")
            if p is None:
                raise KeyError(
                    "Expected proprio_obs in infos for visual render. "
                    "Ensure render env has emit_proprio_info=True."
                )
            p = np.asarray(p, dtype=np.float32)
            out[agent] = np.concatenate([p, z], axis=-1).astype(np.float32)
        else:
            out[agent] = z
    return out


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    train_args = ckpt.get("args", {})
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    hidden_size = int(train_args.get("hidden_size", 256))
    adapter = _build_obs_adapter(train_args=train_args, cli_args=args, device=torch.device("cpu"))

    actors = {agent: Actor(obs_dim, act_dim, hidden_size) for agent in AGENTS}
    for agent in AGENTS:
        actors[agent].load_state_dict(ckpt["actors"][agent])
        actors[agent].eval()

    env = make_env_from_ckpt_args(train_args, args)
    env_act_dim = int(env.action_space("h0").shape[0])
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
            policy_obs = _transform_obs(obs, infos, adapter)
            env_obs_dim = int(np.asarray(policy_obs["h0"]).shape[0])
            if env_obs_dim != obs_dim:
                raise ValueError(
                    f"Observation-dimension mismatch between checkpoint and env: "
                    f"ckpt obs_dim={obs_dim}, env obs_dim={env_obs_dim}. "
                    "Use matching env configuration for rendering."
                )
            done = False
            ep_ret = 0.0
            steps = 0
            reason = "running"

            while not done:
                actions = {}
                for agent in AGENTS:
                    obs_t = torch.as_tensor(policy_obs[agent], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = actors[agent](obs_t)
                    act_np = action.squeeze(0).cpu().numpy()
                    if args.noise > 0.0:
                        act_np = act_np + rng.normal(0.0, args.noise, size=act_np.shape)
                    actions[agent] = np.clip(act_np, -1.0, 1.0).astype(np.float32)

                obs, rewards, terminations, truncations, infos = env.step(actions)
                policy_obs = _transform_obs(obs, infos, adapter)
                env.render()
                time.sleep(HUMAN_RENDER_DELAY_S)
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
