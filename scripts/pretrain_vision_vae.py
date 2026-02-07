"""Pretrain a visual VAE on random HumanoidCollab rollouts."""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "VAE pretraining requires PyTorch. Install training dependencies with: "
        "pip install -e '.[train]'"
    ) from exc

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.tasks.registry import available_tasks
from humanoid_collab.vision import ConvVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a visual VAE for HumanoidCollab observations.")
    parser.add_argument("--task", type=str, default="handshake", choices=available_tasks())
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hold-target", type=int, default=30)
    parser.add_argument(
        "--fixed-standing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Weld torsos to world; keep False for locomotion data.",
    )
    parser.add_argument("--control-mode", type=str, default="all", choices=["all", "arms_only"])
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="gray",
        choices=["gray", "rgb"],
        help="Visual mode used to collect frames.",
    )
    parser.add_argument("--obs-rgb-width", type=int, default=84)
    parser.add_argument("--obs-rgb-height", type=int, default=84)
    parser.add_argument("--physics-profile", type=str, default="default", choices=available_physics_profiles())
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-frames", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1e-3, help="KL coefficient.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--output", type=str, default="checkpoints/vision_vae_handshake_gray.pt")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_random_frames(args: argparse.Namespace) -> np.ndarray:
    env = HumanoidCollabEnv(
        task=args.task,
        stage=int(args.stage),
        horizon=int(args.horizon),
        frame_skip=int(args.frame_skip),
        hold_target=int(args.hold_target),
        physics_profile=args.physics_profile,
        fixed_standing=bool(args.fixed_standing),
        control_mode=args.control_mode,
        observation_mode=args.observation_mode,
        obs_rgb_width=int(args.obs_rgb_width),
        obs_rgb_height=int(args.obs_rgb_height),
    )

    rng = np.random.default_rng(args.seed)
    frames: List[np.ndarray] = []
    obs, _ = env.reset(seed=args.seed)
    try:
        while len(frames) < args.num_frames:
            for agent in env.possible_agents:
                frames.append(np.asarray(obs[agent], dtype=np.uint8))
                if len(frames) >= args.num_frames:
                    break
            if len(frames) >= args.num_frames:
                break
            if not env.agents:
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                continue
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)
    finally:
        env.close()

    return np.stack(frames, axis=0)


def train_vae(
    frames_u8: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[ConvVAE, dict]:
    if frames_u8.ndim != 4:
        raise ValueError(f"Expected frame tensor shape (N,H,W,C), got {frames_u8.shape}")
    channels = int(frames_u8.shape[-1])
    vae = ConvVAE(in_channels=channels, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # Normalize once and keep on CPU for minibatching.
    x = torch.as_tensor(frames_u8, dtype=torch.float32) / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    history = {"loss": [], "recon": [], "kl": []}
    vae.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            recon, mu, logvar = vae(batch_x)
            recon_loss = F.mse_loss(recon, batch_x, reduction="mean")
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + args.beta * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = int(batch_x.shape[0])
            loss_sum += float(loss.item()) * bs
            recon_sum += float(recon_loss.item()) * bs
            kl_sum += float(kl.item()) * bs
            count += bs

        mean_loss = loss_sum / max(1, count)
        mean_recon = recon_sum / max(1, count)
        mean_kl = kl_sum / max(1, count)
        history["loss"].append(mean_loss)
        history["recon"].append(mean_recon)
        history["kl"].append(mean_kl)
        if (epoch + 1) % args.log_every == 0:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"loss={mean_loss:.6f} recon={mean_recon:.6f} kl={mean_kl:.6f}"
            )

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, history


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(
        "collecting frames: "
        f"task={args.task} stage={args.stage} mode={args.observation_mode} num_frames={args.num_frames}"
    )
    frames_u8 = collect_random_frames(args)
    print(f"collected shape={tuple(frames_u8.shape)}")

    vae, history = train_vae(frames_u8, args=args, device=device)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "algo": "vision_vae",
        "task": args.task,
        "stage": int(args.stage),
        "observation_mode": args.observation_mode,
        "obs_shape": tuple(int(v) for v in frames_u8.shape[1:]),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "beta": float(args.beta),
        "state_dict": vae.state_dict(),
        "history": history,
    }
    torch.save(payload, args.output)
    print(f"saved VAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
