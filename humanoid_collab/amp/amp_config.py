"""Configuration dataclass for AMP training."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class AMPConfig:
    """Configuration for Adversarial Motion Priors training.

    Attributes:
        motion_data_dir: Path to processed motion data directory
        disc_hidden_sizes: Hidden layer sizes for discriminator MLP
        disc_lr: Learning rate for discriminator
        disc_batch_size: Batch size for discriminator training
        n_disc_updates: Number of discriminator updates per policy update
        lambda_gp: Gradient penalty coefficient (WGAN-GP)
        reward_scale: Scale factor for AMP reward
        clip_reward: Whether to use clipped stable reward form
        amp_weight: Initial weight for AMP reward in combined objective
        alive_bonus: Bonus for staying upright (pre-training only)
        energy_penalty: Penalty coefficient for control effort
    """
    # Motion data
    motion_data_dir: str = "data/amass/processed"
    motion_categories: Tuple[str, ...] = ("standing", "walking", "reaching")

    # Discriminator architecture
    disc_hidden_sizes: Tuple[int, ...] = (1024, 512)
    disc_activation: str = "relu"

    # Discriminator training
    disc_lr: float = 1e-4
    disc_batch_size: int = 256
    n_disc_updates: int = 5
    lambda_gp: float = 10.0

    # AMP reward
    reward_scale: float = 2.0
    clip_reward: bool = True
    amp_weight: float = 0.5

    # Pre-training bonuses/penalties
    alive_bonus: float = 0.1
    energy_penalty: float = 0.0005
    fall_penalty: float = 10.0

    # Training schedule
    pretrain_steps: int = 1_000_000
    warmup_steps: int = 10_000  # Steps before starting discriminator updates

    # AMP observation settings
    include_root_height: bool = True
    include_root_orientation: bool = True
    include_joint_positions: bool = True
    include_joint_velocities: bool = True

    def get_amp_obs_dim(self, nq_agent: int, nv_agent: int) -> int:
        """Calculate AMP observation dimension based on config.

        Args:
            nq_agent: Number of qpos entries for agent (including 7 for root)
            nv_agent: Number of qvel entries for agent (including 6 for root)

        Returns:
            Total AMP observation dimension
        """
        dim = 0

        if self.include_joint_positions:
            dim += nq_agent - 7  # Exclude root position + quaternion

        if self.include_joint_velocities:
            dim += nv_agent - 6  # Exclude root linear + angular velocity

        if self.include_root_height:
            dim += 1

        if self.include_root_orientation:
            dim += 6  # Forward + up vectors (3 + 3)
            dim += 3  # Root angular velocity

        return dim


@dataclass
class AMPTrainState:
    """Tracks AMP training progress.

    Attributes:
        total_steps: Total environment steps taken
        disc_updates: Number of discriminator updates performed
        phase: Current training phase ("pretrain", "approach", "task")
        amp_weight: Current AMP reward weight
    """
    total_steps: int = 0
    disc_updates: int = 0
    phase: str = "pretrain"
    amp_weight: float = 0.5

    # Metrics tracking
    disc_loss_ema: float = 0.0
    disc_real_ema: float = 0.0
    disc_fake_ema: float = 0.0
    gp_loss_ema: float = 0.0
    amp_reward_ema: float = 0.0
