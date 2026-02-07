"""Vision models and helpers for image-based policies."""

from humanoid_collab.vision.vae import ConvVAE, load_frozen_encoder_from_vae_checkpoint

__all__ = ["ConvVAE", "load_frozen_encoder_from_vae_checkpoint"]
