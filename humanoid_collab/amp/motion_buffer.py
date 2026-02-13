"""Motion replay buffer for storing reference motion observations.

Pre-computes AMP observations from motion clips for efficient discriminator training.
"""

from typing import Tuple, Optional
import numpy as np
import torch

from humanoid_collab.amp.motion_data import MotionDataset
from humanoid_collab.amp.amp_obs import AMPObsBuilder, quat_to_mat


class MotionReplayBuffer:
    """Stores pre-computed AMP observations from reference motion clips.

    Pre-computes all AMP observations from motion clips during initialization
    to enable fast sampling during training.
    """

    def __init__(
        self,
        dataset: MotionDataset,
        amp_obs_builder: AMPObsBuilder,
    ):
        """Initialize the motion replay buffer.

        Args:
            dataset: Collection of motion clips
            amp_obs_builder: Builder for AMP observations
        """
        self.dataset = dataset
        self.amp_obs_builder = amp_obs_builder

        # Pre-compute all observations
        self._precompute_observations()

    def _precompute_observations(self) -> None:
        """Pre-compute AMP observations for all motion transitions."""
        obs_t_list = []
        obs_t1_list = []

        for clip in self.dataset.clips:
            # Compute observations for each frame
            for t in range(clip.num_frames - 1):
                qpos_t = clip.qpos[t]
                qvel_t = clip.qvel[t]
                qpos_t1 = clip.qpos[t + 1]
                qvel_t1 = clip.qvel[t + 1]

                # Compute rotation matrices from quaternions
                xmat_t = quat_to_mat(qpos_t[3:7])
                xmat_t1 = quat_to_mat(qpos_t1[3:7])

                # Compute AMP observations
                obs_t = self.amp_obs_builder.compute_obs_from_qpos_qvel(
                    qpos_t, qvel_t, xmat_t
                )
                obs_t1 = self.amp_obs_builder.compute_obs_from_qpos_qvel(
                    qpos_t1, qvel_t1, xmat_t1
                )

                obs_t_list.append(obs_t)
                obs_t1_list.append(obs_t1)

        self.obs_t = np.stack(obs_t_list).astype(np.float32)
        self.obs_t1 = np.stack(obs_t1_list).astype(np.float32)

        self._num_transitions = len(self.obs_t)
        self._rng = np.random.default_rng()

    @property
    def num_transitions(self) -> int:
        """Total number of stored transitions."""
        return self._num_transitions

    @property
    def obs_dim(self) -> int:
        """AMP observation dimension."""
        return self.obs_t.shape[1]

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of reference motion transitions.

        Args:
            batch_size: Number of transitions to sample
            rng: Random number generator (uses internal if None)

        Returns:
            Tuple of (obs_t, obs_t1) with shape (batch_size, obs_dim)
        """
        if rng is None:
            rng = self._rng

        indices = rng.integers(0, self._num_transitions, size=batch_size)
        return self.obs_t[indices], self.obs_t1[indices]

    def sample_torch(
        self,
        batch_size: int,
        device: str = "cpu",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch as PyTorch tensors.

        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            rng: Random number generator

        Returns:
            Tuple of (obs_t, obs_t1) as tensors
        """
        obs_t, obs_t1 = self.sample(batch_size, rng)
        return (
            torch.as_tensor(obs_t, device=device),
            torch.as_tensor(obs_t1, device=device),
        )


class PolicyTransitionBuffer:
    """Buffer for storing policy-generated transitions during rollout.

    Used to collect fake samples for discriminator training.
    """

    def __init__(self, capacity: int, obs_dim: int):
        """Initialize the buffer.

        Args:
            capacity: Maximum number of transitions to store
            obs_dim: AMP observation dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.obs_t = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs_t1 = np.zeros((capacity, obs_dim), dtype=np.float32)

        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng()

    def add(self, obs_t: np.ndarray, obs_t1: np.ndarray) -> None:
        """Add a transition to the buffer.

        Args:
            obs_t: AMP observation at time t
            obs_t1: AMP observation at time t+1
        """
        self.obs_t[self._ptr] = obs_t
        self.obs_t1[self._ptr] = obs_t1

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, obs_t: np.ndarray, obs_t1: np.ndarray) -> None:
        """Add a batch of transitions to the buffer.

        Args:
            obs_t: AMP observations at time t (batch, obs_dim)
            obs_t1: AMP observations at time t+1 (batch, obs_dim)
        """
        batch_size = len(obs_t)

        for i in range(batch_size):
            self.add(obs_t[i], obs_t1[i])

    def clear(self) -> None:
        """Clear the buffer."""
        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            rng: Random number generator

        Returns:
            Tuple of (obs_t, obs_t1)
        """
        if rng is None:
            rng = self._rng

        if self._size == 0:
            raise ValueError("Buffer is empty")

        batch_size = min(batch_size, self._size)
        indices = rng.integers(0, self._size, size=batch_size)

        return self.obs_t[indices], self.obs_t1[indices]

    def sample_torch(
        self,
        batch_size: int,
        device: str = "cpu",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch as PyTorch tensors.

        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            rng: Random number generator

        Returns:
            Tuple of (obs_t, obs_t1) as tensors
        """
        obs_t, obs_t1 = self.sample(batch_size, rng)
        return (
            torch.as_tensor(obs_t, device=device),
            torch.as_tensor(obs_t1, device=device),
        )

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all stored transitions.

        Returns:
            Tuple of (obs_t, obs_t1) with current buffer contents
        """
        return self.obs_t[:self._size], self.obs_t1[:self._size]

    def get_all_torch(
        self,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all stored transitions as PyTorch tensors.

        Args:
            device: Device to place tensors on

        Returns:
            Tuple of (obs_t, obs_t1) as tensors
        """
        obs_t, obs_t1 = self.get_all()
        return (
            torch.as_tensor(obs_t, device=device),
            torch.as_tensor(obs_t1, device=device),
        )
