"""Convolutional VAE used for frozen latent encoding in RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Compact convolutional VAE for egocentric humanoid observations."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_pool_hw = (4, 4)
        enc_flat = 128 * self.encoder_pool_hw[0] * self.encoder_pool_hw[1]
        self.encoder_mlp = nn.Sequential(
            nn.Linear(enc_flat, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, enc_flat),
            nn.ReLU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=3, stride=1, padding=1),
        )

    def _encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = F.adaptive_avg_pool2d(h, self.encoder_pool_hw)
        h = h.flatten(start_dim=1)
        return self.encoder_mlp(h)

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(mu, logvar)` for a normalized BCHW input in `[0, 1]`."""
        h = self._encode_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent mean for a normalized BCHW input."""
        mu, _ = self.encode_stats(x)
        return mu

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, output_hw: Tuple[int, int]) -> torch.Tensor:
        h = self.decoder_mlp(z)
        h = h.view(z.shape[0], 128, self.encoder_pool_hw[0], self.encoder_pool_hw[1])
        x_hat = self.decoder_cnn(h)
        if tuple(x_hat.shape[-2:]) != tuple(output_hw):
            x_hat = F.interpolate(x_hat, size=output_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(x_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with normalized BCHW input in `[0, 1]`."""
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, output_hw=tuple(x.shape[-2:]))
        return recon, mu, logvar


@dataclass
class FrozenEncoderBundle:
    """Loaded frozen encoder and metadata from a VAE checkpoint."""

    vae: ConvVAE
    observation_mode: str
    obs_shape: Tuple[int, int, int]
    latent_dim: int
    checkpoint_payload: Dict[str, object]


def load_frozen_encoder_from_vae_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> FrozenEncoderBundle:
    """Load a VAE checkpoint and return a frozen encoder bundle."""
    payload = torch.load(checkpoint_path, map_location=device)
    algo = str(payload.get("algo", ""))
    if algo != "vision_vae":
        raise ValueError(
            f"Checkpoint at '{checkpoint_path}' is not a VAE checkpoint "
            f"(expected algo='vision_vae', got '{algo}')."
        )
    observation_mode = str(payload["observation_mode"])
    obs_shape = tuple(int(v) for v in payload["obs_shape"])
    if len(obs_shape) != 3:
        raise ValueError("VAE checkpoint obs_shape must be (H, W, C).")
    latent_dim = int(payload["latent_dim"])
    in_channels = int(obs_shape[2])
    hidden_dim = int(payload.get("hidden_dim", 256))
    vae = ConvVAE(in_channels=in_channels, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    vae.load_state_dict(payload["state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return FrozenEncoderBundle(
        vae=vae,
        observation_mode=observation_mode,
        obs_shape=obs_shape,
        latent_dim=latent_dim,
        checkpoint_payload=payload,
    )
