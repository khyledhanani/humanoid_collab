"""Observation builder for constructing per-agent observation vectors.

Builds base observations with strictly self/egocentric proprioceptive
features only. Task-specific observations are appended by the task config.
"""

from typing import Dict
import numpy as np
import mujoco

from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    get_root_angular_velocity,
)


class ObservationBuilder:
    """Builds per-agent base observation vectors."""

    def __init__(self, id_cache: IDCache, model: mujoco.MjModel, task_obs_dim: int = 0):
        """Initialize the observation builder.

        Args:
            id_cache: Cached IDs from the MuJoCo model
            model: MuJoCo model instance
            task_obs_dim: Number of additional task-specific observation dimensions
        """
        self.id_cache = id_cache
        self.model = model
        self.agents = ["h0", "h1"]
        self.task_obs_dim = task_obs_dim

        self._compute_obs_dim()

    def _compute_obs_dim(self) -> None:
        """Compute the total observation dimension."""
        nq_agent = len(self.id_cache.joint_qpos_idx["h0"])
        nv_agent = len(self.id_cache.joint_qvel_idx["h0"])

        # Proprioception: joint pos/vel + root quat + root angvel
        proprio_dim = (nq_agent - 7) + (nv_agent - 6) + 4 + 3

        self.base_obs_dim = proprio_dim
        self.obs_dim = self.base_obs_dim + self.task_obs_dim

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_base_obs_dim(self) -> int:
        return self.base_obs_dim

    def build_base_observations(
        self,
        data: mujoco.MjData,
    ) -> Dict[str, np.ndarray]:
        """Build base observation vectors (without task-specific parts).

        Args:
            data: MuJoCo data instance

        Returns:
            Dictionary mapping agent ID to base observation array
        """
        obs = {}
        for agent in self.agents:
            obs[agent] = self._build_agent_base_obs(data, agent)
        return obs

    def _build_agent_base_obs(
        self,
        data: mujoco.MjData,
        agent: str,
    ) -> np.ndarray:
        """Build base observation vector for a single agent."""
        obs_parts = []

        # === Proprioception ===
        qpos_idx = self.id_cache.joint_qpos_idx[agent]
        qvel_idx = self.id_cache.joint_qvel_idx[agent]

        # Joint positions (excluding root pos+quat)
        joint_qpos = data.qpos[qpos_idx[7:]]
        obs_parts.append(joint_qpos)

        # Joint velocities (excluding root lin+ang vel)
        joint_qvel = data.qvel[qvel_idx[6:]]
        obs_parts.append(joint_qvel)

        # Root orientation (quaternion)
        root_quat = data.qpos[qpos_idx[3:7]]
        obs_parts.append(root_quat)

        # Root angular velocity
        root_angvel = get_root_angular_velocity(data, qvel_idx)
        obs_parts.append(root_angvel)

        return np.concatenate(obs_parts).astype(np.float32)
