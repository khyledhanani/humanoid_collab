"""AMP observation builder for discriminator input.

Builds style-only observations for the discriminator - these should capture
motion quality without any task-specific or partner-relative features.
"""

from typing import Dict, Tuple
import numpy as np
import mujoco

from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    get_forward_vector,
    get_up_vector,
    get_root_angular_velocity,
)


class AMPObsBuilder:
    """Builds discriminator observations from MuJoCo state.

    Observation features (per agent):
    - Joint positions in root frame (excluding root pos+quat): nq - 7
    - Joint velocities (excluding root lin+ang vel): nv - 6
    - Root height: 1
    - Root forward vector: 3
    - Root up vector: 3
    - Root angular velocity: 3

    Total: (nq - 7) + (nv - 6) + 1 + 3 + 3 + 3 = nq + nv - 3
    """

    def __init__(
        self,
        id_cache: IDCache,
        include_root_height: bool = True,
        include_root_orientation: bool = True,
        include_joint_positions: bool = True,
        include_joint_velocities: bool = True,
    ):
        """Initialize the AMP observation builder.

        Args:
            id_cache: Cached IDs from the MuJoCo model
            include_root_height: Include root height in observations
            include_root_orientation: Include root orientation vectors
            include_joint_positions: Include joint positions
            include_joint_velocities: Include joint velocities
        """
        self.id_cache = id_cache
        self.agents = ["h0", "h1"]

        self.include_root_height = include_root_height
        self.include_root_orientation = include_root_orientation
        self.include_joint_positions = include_joint_positions
        self.include_joint_velocities = include_joint_velocities

        self._compute_obs_dim()

    def _compute_obs_dim(self) -> None:
        """Compute the AMP observation dimension."""
        nq_agent = len(self.id_cache.joint_qpos_idx["h0"])
        nv_agent = len(self.id_cache.joint_qvel_idx["h0"])

        dim = 0

        if self.include_joint_positions:
            dim += nq_agent - 7  # Exclude root position + quaternion

        if self.include_joint_velocities:
            dim += nv_agent - 6  # Exclude root linear + angular velocity

        if self.include_root_height:
            dim += 1

        if self.include_root_orientation:
            dim += 3  # Forward vector
            dim += 3  # Up vector
            dim += 3  # Angular velocity

        self._obs_dim = dim

    @property
    def obs_dim(self) -> int:
        """Total observation dimension per agent."""
        return self._obs_dim

    def compute_obs(
        self,
        data: mujoco.MjData,
        agent: str,
    ) -> np.ndarray:
        """Compute AMP observation for a single agent.

        Args:
            data: MuJoCo data instance
            agent: Agent ID ("h0" or "h1")

        Returns:
            AMP observation vector
        """
        obs_parts = []

        qpos_idx = self.id_cache.joint_qpos_idx[agent]
        qvel_idx = self.id_cache.joint_qvel_idx[agent]

        # Joint positions (excluding root)
        if self.include_joint_positions:
            joint_qpos = data.qpos[qpos_idx[7:]]
            obs_parts.append(joint_qpos)

        # Joint velocities (excluding root)
        if self.include_joint_velocities:
            joint_qvel = data.qvel[qvel_idx[6:]]
            obs_parts.append(joint_qvel)

        # Root height
        if self.include_root_height:
            root_height = data.qpos[qpos_idx[2]]  # z position
            obs_parts.append(np.array([root_height]))

        # Root orientation
        if self.include_root_orientation:
            xmat = self.id_cache.get_torso_xmat(data, agent)

            # Forward vector
            fwd = get_forward_vector(xmat)
            obs_parts.append(fwd)

            # Up vector
            up = get_up_vector(xmat)
            obs_parts.append(up)

            # Angular velocity
            angvel = get_root_angular_velocity(data, qvel_idx)
            obs_parts.append(angvel)

        return np.concatenate(obs_parts).astype(np.float32)

    def compute_obs_both(
        self,
        data: mujoco.MjData,
    ) -> Dict[str, np.ndarray]:
        """Compute AMP observations for both agents.

        Args:
            data: MuJoCo data instance

        Returns:
            Dictionary mapping agent ID to observation
        """
        return {agent: self.compute_obs(data, agent) for agent in self.agents}

    def compute_obs_from_qpos_qvel(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        xmat: np.ndarray,
    ) -> np.ndarray:
        """Compute AMP observation from raw qpos/qvel arrays.

        This is used for computing observations from motion clip data
        without a full MuJoCo simulation.

        Args:
            qpos: Joint positions (nq,) for one agent
            qvel: Joint velocities (nv,) for one agent
            xmat: Root rotation matrix (3, 3)

        Returns:
            AMP observation vector
        """
        obs_parts = []

        # Joint positions (excluding root)
        if self.include_joint_positions:
            joint_qpos = qpos[7:]
            obs_parts.append(joint_qpos)

        # Joint velocities (excluding root)
        if self.include_joint_velocities:
            joint_qvel = qvel[6:]
            obs_parts.append(joint_qvel)

        # Root height
        if self.include_root_height:
            root_height = qpos[2]
            obs_parts.append(np.array([root_height]))

        # Root orientation
        if self.include_root_orientation:
            # Forward vector
            fwd = get_forward_vector(xmat)
            obs_parts.append(fwd)

            # Up vector
            up = get_up_vector(xmat)
            obs_parts.append(up)

            # Angular velocity
            angvel = qvel[3:6]
            obs_parts.append(angvel)

        return np.concatenate(obs_parts).astype(np.float32)

    def compute_obs_batch_from_clips(
        self,
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
    ) -> np.ndarray:
        """Compute AMP observations for a batch of motion clip frames.

        Args:
            qpos_batch: Joint positions (batch, nq)
            qvel_batch: Joint velocities (batch, nv)

        Returns:
            AMP observations (batch, obs_dim)
        """
        batch_size = len(qpos_batch)
        obs_batch = np.zeros((batch_size, self.obs_dim), dtype=np.float32)

        for i in range(batch_size):
            # Compute rotation matrix from quaternion
            quat = qpos_batch[i, 3:7]
            xmat = quat_to_mat(quat)

            obs_batch[i] = self.compute_obs_from_qpos_qvel(
                qpos_batch[i], qvel_batch[i], xmat
            )

        return obs_batch


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        quat: Quaternion as (w, x, y, z)

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-8:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
