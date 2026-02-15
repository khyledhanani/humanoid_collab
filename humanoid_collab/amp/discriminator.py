"""Discriminator for Adversarial Motion Priors.

The discriminator learns to distinguish between reference motion transitions
(from motion capture data) and policy-generated transitions.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class AMPDiscriminator(nn.Module):
    """Discriminator for motion quality scoring.

    Takes state transitions (s_t, s_{t+1}) as input and outputs a scalar
    "realness" score. Higher scores indicate more natural motion.

    The trainer supports both:
    - ``wgan_gp``: maximize D(real) - D(fake)
    - ``lsgan_gp``: least-squares real/fake targets (+1 / -1)
    with gradient penalty on interpolated samples.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (1024, 512),
        activation: str = "relu",
    ):
        """Initialize the discriminator.

        Args:
            obs_dim: Dimension of AMP observation per timestep
            hidden_sizes: Hidden layer sizes for MLP
            activation: Activation function ("relu" or "elu")
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.input_dim = obs_dim * 2  # Concatenated (s_t, s_{t+1})

        # Build MLP
        layers = []
        prev_dim = self.input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = h

        # Output layer (no activation - raw score)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of state transitions.

        Args:
            obs_t: AMP observations at time t (batch, obs_dim)
            obs_t1: AMP observations at time t+1 (batch, obs_dim)

        Returns:
            Discriminator scores (batch,)
        """
        x = torch.cat([obs_t, obs_t1], dim=-1)
        return self.net(x).squeeze(-1)

    def compute_reward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        clip_reward: bool = True,
        reward_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute AMP style reward from discriminator.

        Args:
            obs_t: AMP observations at time t (batch, obs_dim)
            obs_t1: AMP observations at time t+1 (batch, obs_dim)
            clip_reward: Use clipped stable reward form
            reward_scale: Scale factor for reward

        Returns:
            Style rewards (batch,)
        """
        with torch.no_grad():
            d_score = self.forward(obs_t, obs_t1)

            if clip_reward:
                # Stable squared form: max(0, 1 - 0.25 * (1 - D(s,s'))^2)
                reward = torch.clamp(1.0 - 0.25 * (1.0 - d_score) ** 2, min=0.0)
            else:
                # Original log form: -log(1 - sigmoid(D(s,s')))
                prob = torch.sigmoid(d_score)
                reward = -torch.log(torch.clamp(1.0 - prob, min=1e-6))

            return reward_scale * reward


def compute_gradient_penalty(
    discriminator: AMPDiscriminator,
    real_obs_t: torch.Tensor,
    real_obs_t1: torch.Tensor,
    fake_obs_t: torch.Tensor,
    fake_obs_t1: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute gradient penalty for AMP discriminator training.

    Interpolates between real and fake samples and penalizes gradients
    that deviate from unit norm.

    Args:
        discriminator: Discriminator network
        real_obs_t: Real observations at time t
        real_obs_t1: Real observations at time t+1
        fake_obs_t: Fake observations at time t
        fake_obs_t1: Fake observations at time t+1
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty loss
    """
    batch_size = real_obs_t.shape[0]
    device = real_obs_t.device

    # Random interpolation weight
    alpha = torch.rand(batch_size, 1, device=device)

    # Interpolate between real and fake
    interp_t = alpha * real_obs_t + (1 - alpha) * fake_obs_t
    interp_t1 = alpha * real_obs_t1 + (1 - alpha) * fake_obs_t1

    interp_t.requires_grad_(True)
    interp_t1.requires_grad_(True)

    # Get discriminator output on interpolated samples
    d_interp = discriminator(interp_t, interp_t1)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=[interp_t, interp_t1],
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )

    # Compute gradient norm
    grad_t, grad_t1 = gradients
    grad_norm = torch.sqrt(
        grad_t.pow(2).sum(dim=-1) + grad_t1.pow(2).sum(dim=-1) + 1e-8
    )

    # Penalize deviation from unit norm
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


class AMPDiscriminatorTrainer:
    """Handles discriminator training with configurable objective."""

    def __init__(
        self,
        discriminator: AMPDiscriminator,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.5, 0.999),
        lambda_gp: float = 10.0,
        n_updates: int = 5,
        objective: str = "wgan_gp",
        device: str = "cpu",
    ):
        """Initialize the trainer.

        Args:
            discriminator: Discriminator network
            lr: Learning rate
            betas: Adam betas
            lambda_gp: Gradient penalty coefficient
            n_updates: Number of discriminator updates per batch
            objective: One of {"wgan_gp", "lsgan_gp"}
            device: Device to train on
        """
        if objective not in {"wgan_gp", "lsgan_gp"}:
            raise ValueError(f"Unknown discriminator objective '{objective}'")

        self.discriminator = discriminator.to(device)
        self.device = device
        self.lambda_gp = lambda_gp
        self.n_updates = n_updates
        self.objective = objective

        self.optimizer = optim.Adam(
            discriminator.parameters(),
            lr=lr,
            betas=betas,
        )

    def train_step(
        self,
        real_obs_t: torch.Tensor,
        real_obs_t1: torch.Tensor,
        fake_obs_t: torch.Tensor,
        fake_obs_t1: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform discriminator training step.

        Args:
            real_obs_t: Real observations at time t (batch, obs_dim)
            real_obs_t1: Real observations at time t+1 (batch, obs_dim)
            fake_obs_t: Fake observations at time t (batch, obs_dim)
            fake_obs_t1: Fake observations at time t+1 (batch, obs_dim)

        Returns:
            Dictionary of training metrics
        """
        real_obs_t = real_obs_t.to(self.device)
        real_obs_t1 = real_obs_t1.to(self.device)
        fake_obs_t = fake_obs_t.to(self.device)
        fake_obs_t1 = fake_obs_t1.to(self.device)

        metrics = {
            "disc_loss": 0.0,
            "gp_loss": 0.0,
            "d_real": 0.0,
            "d_fake": 0.0,
            "real_loss": 0.0,
            "fake_loss": 0.0,
        }

        for _ in range(self.n_updates):
            self.optimizer.zero_grad()

            # Discriminator scores
            d_real = self.discriminator(real_obs_t, real_obs_t1)
            d_fake = self.discriminator(fake_obs_t, fake_obs_t1)

            if self.objective == "lsgan_gp":
                real_loss = 0.5 * ((d_real - 1.0) ** 2).mean()
                fake_loss = 0.5 * ((d_fake + 1.0) ** 2).mean()
                disc_loss = real_loss + fake_loss
            else:
                # WGAN loss: maximize D(real) - D(fake)
                # Equivalently, minimize D(fake) - D(real)
                real_loss = torch.zeros((), device=self.device)
                fake_loss = torch.zeros((), device=self.device)
                disc_loss = d_fake.mean() - d_real.mean()

            # Gradient penalty
            gp_loss = compute_gradient_penalty(
                self.discriminator,
                real_obs_t, real_obs_t1,
                fake_obs_t, fake_obs_t1,
                self.lambda_gp,
            )

            # Total loss
            total_loss = disc_loss + gp_loss
            total_loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            metrics["disc_loss"] += disc_loss.item() / self.n_updates
            metrics["gp_loss"] += gp_loss.item() / self.n_updates
            metrics["d_real"] += d_real.mean().item() / self.n_updates
            metrics["d_fake"] += d_fake.mean().item() / self.n_updates
            metrics["real_loss"] += real_loss.item() / self.n_updates
            metrics["fake_loss"] += fake_loss.item() / self.n_updates

        return metrics

    def save(self, path: str) -> None:
        """Save discriminator and optimizer state."""
        torch.save({
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load discriminator and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
