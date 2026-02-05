"""Observation builder for constructing per-agent observation vectors.

Builds base observations (proprioception + partner relative features)
shared across all tasks. Task-specific observations are appended by
the task config.
"""

from typing import Dict
import numpy as np
import mujoco

from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    get_forward_vector,
    get_up_vector,
    rotate_to_local_frame,
    compute_facing_alignment,
    compute_up_alignment,
    get_root_linear_velocity,
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

        # Partner relative features: chest(3) + pelvis(3) + velocity(3) + facing(1) + up(1)
        partner_dim = 3 + 3 + 3 + 1 + 1

        self.base_obs_dim = proprio_dim + partner_dim
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
        partner = "h1" if agent == "h0" else "h0"
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

        # === Partner relative features ===
        self_xpos = self.id_cache.get_torso_xpos(data, agent)
        self_xmat = self.id_cache.get_torso_xmat(data, agent)
        partner_xpos = self.id_cache.get_torso_xpos(data, partner)
        partner_xmat = self.id_cache.get_torso_xmat(data, partner)

        # Relative position of partner chest (in local frame)
        try:
            partner_chest = self.id_cache.get_site_xpos(data, f"{partner}_chest")
        except ValueError:
            partner_chest = partner_xpos
        rel_chest = partner_chest - self_xpos
        rel_chest_local = rotate_to_local_frame(rel_chest, self_xmat)
        obs_parts.append(rel_chest_local)

        # Relative position of partner pelvis (in local frame)
        try:
            partner_pelvis = self.id_cache.get_site_xpos(data, f"{partner}_pelvis")
        except ValueError:
            partner_pelvis = partner_xpos - np.array([0, 0, 0.2])
        rel_pelvis = partner_pelvis - self_xpos
        rel_pelvis_local = rotate_to_local_frame(rel_pelvis, self_xmat)
        obs_parts.append(rel_pelvis_local)

        # Relative velocity of partner root (in local frame)
        partner_qvel_idx = self.id_cache.joint_qvel_idx[partner]
        partner_linvel = get_root_linear_velocity(data, partner_qvel_idx)
        self_linvel = get_root_linear_velocity(data, qvel_idx)
        rel_vel = partner_linvel - self_linvel
        rel_vel_local = rotate_to_local_frame(rel_vel, self_xmat)
        obs_parts.append(rel_vel_local)

        # Facing alignment
        self_fwd = get_forward_vector(self_xmat)
        partner_fwd = get_forward_vector(partner_xmat)
        facing = compute_facing_alignment(self_fwd, partner_fwd)
        obs_parts.append(np.array([facing]))

        # Up alignment
        self_up = get_up_vector(self_xmat)
        partner_up = get_up_vector(partner_xmat)
        up_align = compute_up_alignment(self_up, partner_up)
        obs_parts.append(np.array([up_align]))

        return np.concatenate(obs_parts).astype(np.float32)
