"""Generalized ID caching utilities for efficient MuJoCo lookups."""

from typing import Dict, Set, List, Optional
import numpy as np
import mujoco


class IDCache:
    """Caches MuJoCo IDs for sites, geoms, bodies, joints, and actuators.

    Supports dynamic registration of custom geom groups for task-specific
    contact detection.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.agents = ["h0", "h1"]

        # Site IDs
        self.site_ids: Dict[str, int] = {}

        # Standard humanoid geom groups
        self.arm_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.torso_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.left_arm_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.right_arm_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.hand_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.left_hand_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.right_hand_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}

        # Custom geom groups (registered by tasks)
        self.geom_groups: Dict[str, Set[int]] = {}

        # Body IDs
        self.body_ids: Dict[str, int] = {}
        self.torso_body_ids: Dict[str, int] = {}

        # Actuator indices per agent
        self.actuator_idx: Dict[str, np.ndarray] = {}

        # Joint indices per agent
        self.joint_qpos_idx: Dict[str, np.ndarray] = {}
        self.joint_qvel_idx: Dict[str, np.ndarray] = {}

        self._cache_all()

    def _cache_all(self) -> None:
        self._cache_sites()
        self._cache_geoms()
        self._cache_bodies()
        self._cache_actuators()
        self._cache_joints()
        self._register_standard_geom_groups()

    def _cache_sites(self) -> None:
        """Cache all site IDs in the model."""
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name is not None:
                self.site_ids[site_name] = i

    def _cache_geoms(self) -> None:
        """Cache geom IDs for arms, hands, and torso of each humanoid."""
        arm_keywords = ["upper_arm", "lower_arm", "hand"]
        left_arm_keywords = ["left_upper_arm", "left_lower_arm", "left_hand"]
        right_arm_keywords = ["right_upper_arm", "right_lower_arm", "right_hand"]
        hand_keywords = ["hand_geom"]
        left_hand_keywords = ["left_hand_geom"]
        right_hand_keywords = ["right_hand_geom"]
        torso_keywords = ["torso_geom", "chest_geom", "abdomen_geom"]

        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name is None:
                continue

            for agent in self.agents:
                if not geom_name.startswith(f"{agent}_"):
                    continue

                for kw in arm_keywords:
                    if kw in geom_name:
                        self.arm_geoms[agent].add(i)
                        break

                for kw in left_arm_keywords:
                    if kw in geom_name:
                        self.left_arm_geoms[agent].add(i)
                        break

                for kw in right_arm_keywords:
                    if kw in geom_name:
                        self.right_arm_geoms[agent].add(i)
                        break

                for kw in hand_keywords:
                    if kw in geom_name:
                        self.hand_geoms[agent].add(i)
                        break

                for kw in left_hand_keywords:
                    if kw in geom_name:
                        self.left_hand_geoms[agent].add(i)
                        break

                for kw in right_hand_keywords:
                    if kw in geom_name:
                        self.right_hand_geoms[agent].add(i)
                        break

                for kw in torso_keywords:
                    if kw in geom_name:
                        self.torso_geoms[agent].add(i)
                        break

    def _cache_bodies(self) -> None:
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                continue
            for agent in self.agents:
                if body_name.startswith(f"{agent}_"):
                    self.body_ids[body_name] = i
                    if body_name == f"{agent}_torso":
                        self.torso_body_ids[agent] = i

    def _cache_actuators(self) -> None:
        for agent in self.agents:
            indices = []
            for i in range(self.model.nu):
                act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if act_name is not None and act_name.startswith(f"{agent}_"):
                    indices.append(i)
            self.actuator_idx[agent] = np.array(indices, dtype=np.int32)

    def _cache_joints(self) -> None:
        for agent in self.agents:
            qpos_indices = []
            qvel_indices = []

            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name is None:
                    continue
                if not joint_name.startswith(f"{agent}_"):
                    continue

                qpos_start = self.model.jnt_qposadr[i]
                qvel_start = self.model.jnt_dofadr[i]
                joint_type = self.model.jnt_type[i]

                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_indices.extend(range(qpos_start, qpos_start + 7))
                    qvel_indices.extend(range(qvel_start, qvel_start + 6))
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    qpos_indices.extend(range(qpos_start, qpos_start + 4))
                    qvel_indices.extend(range(qvel_start, qvel_start + 3))
                else:
                    qpos_indices.append(qpos_start)
                    qvel_indices.append(qvel_start)

            self.joint_qpos_idx[agent] = np.array(qpos_indices, dtype=np.int32)
            self.joint_qvel_idx[agent] = np.array(qvel_indices, dtype=np.int32)

    def _register_standard_geom_groups(self) -> None:
        """Register standard humanoid geom groups for contact detection."""
        for agent in self.agents:
            self.geom_groups[f"{agent}_arm"] = self.arm_geoms[agent]
            self.geom_groups[f"{agent}_torso"] = self.torso_geoms[agent]
            self.geom_groups[f"{agent}_l_arm"] = self.left_arm_geoms[agent]
            self.geom_groups[f"{agent}_r_arm"] = self.right_arm_geoms[agent]
            self.geom_groups[f"{agent}_hand"] = self.hand_geoms[agent]
            self.geom_groups[f"{agent}_l_hand"] = self.left_hand_geoms[agent]
            self.geom_groups[f"{agent}_r_hand"] = self.right_hand_geoms[agent]

    def register_geom_group(self, name: str, geom_ids: Set[int]) -> None:
        """Register a custom geom group for contact detection.

        Args:
            name: Name of the geom group (e.g., 'box')
            geom_ids: Set of geom IDs in this group
        """
        self.geom_groups[name] = geom_ids

    def get_geom_group(self, name: str) -> Set[int]:
        """Get a geom group by name."""
        if name not in self.geom_groups:
            raise ValueError(f"Geom group '{name}' not registered")
        return self.geom_groups[name]

    def get_site_xpos(self, data: mujoco.MjData, site_name: str) -> np.ndarray:
        """Return site position as a read-only view (no copy for speed)."""
        site_id = self.site_ids.get(site_name)
        if site_id is None:
            raise ValueError(f"Site '{site_name}' not found in cache")
        return data.site_xpos[site_id]

    def get_body_xpos(self, data: mujoco.MjData, body_name: str) -> np.ndarray:
        """Return body position as a read-only view (no copy for speed)."""
        body_id = self.body_ids.get(body_name)
        if body_id is None:
            raise ValueError(f"Body '{body_name}' not found in cache")
        return data.xpos[body_id]

    def get_body_xmat(self, data: mujoco.MjData, body_name: str) -> np.ndarray:
        """Return body rotation matrix as a read-only view (no copy for speed)."""
        body_id = self.body_ids.get(body_name)
        if body_id is None:
            raise ValueError(f"Body '{body_name}' not found in cache")
        return data.xmat[body_id].reshape(3, 3)

    def get_torso_xpos(self, data: mujoco.MjData, agent: str) -> np.ndarray:
        """Return torso position as a read-only view (no copy for speed)."""
        body_id = self.torso_body_ids.get(agent)
        if body_id is None:
            raise ValueError(f"Torso body for agent '{agent}' not found")
        return data.xpos[body_id]

    def get_torso_xmat(self, data: mujoco.MjData, agent: str) -> np.ndarray:
        """Return torso rotation matrix as a read-only view (no copy for speed)."""
        body_id = self.torso_body_ids.get(agent)
        if body_id is None:
            raise ValueError(f"Torso body for agent '{agent}' not found")
        return data.xmat[body_id].reshape(3, 3)

    def get_num_actuators(self, agent: str) -> int:
        return len(self.actuator_idx[agent])
