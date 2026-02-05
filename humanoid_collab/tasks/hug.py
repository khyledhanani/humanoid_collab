"""Hug task: Two agents approach, embrace, and hold a stable hug."""

from typing import Dict, List, Tuple, Any
import numpy as np
import mujoco

from humanoid_collab.tasks.base import TaskConfig
from humanoid_collab.tasks.registry import register_task
from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    get_forward_vector,
    get_up_vector,
    compute_facing_alignment,
    compute_tilt_angle,
    get_root_linear_velocity,
)


# Curriculum weights per stage
_HUG_STAGES = {
    0: dict(w_dist=2.0, w_face=1.0, w_stab=0.5, w_contact=0.0, w_hand=0.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=100.0),
    1: dict(w_dist=1.5, w_face=1.0, w_stab=0.5, w_contact=0.0, w_hand=2.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=100.0),
    2: dict(w_dist=1.0, w_face=0.8, w_stab=0.5, w_contact=3.0, w_hand=1.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=150.0),
    3: dict(w_dist=0.5, w_face=0.5, w_stab=0.3, w_contact=5.0, w_hand=0.5,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=200.0),
}

# Hug condition thresholds
D_MIN, D_MAX = 0.25, 0.60
FACING_THRESH = 0.6
V_THRESH = 1.0
TILT_THRESH = 0.5
ALPHA_DIST = 3.0
BETA_SPEED = 2.0


@register_task
class HugTask(TaskConfig):

    @property
    def name(self) -> str:
        return "hug"

    @property
    def num_curriculum_stages(self) -> int:
        return 4

    def __init__(self):
        self._weights = _HUG_STAGES[0].copy()
        self._stage = 0

    def get_contact_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("h0_arm_h1_torso", "h0_arm", "h1_torso"),
            ("h1_arm_h0_torso", "h1_arm", "h0_torso"),
            ("h0_l_arm_h1_torso", "h0_l_arm", "h1_torso"),
            ("h0_r_arm_h1_torso", "h0_r_arm", "h1_torso"),
            ("h1_l_arm_h0_torso", "h1_l_arm", "h0_torso"),
            ("h1_r_arm_h0_torso", "h1_r_arm", "h0_torso"),
        ]

    def randomize_state(self, model, data, id_cache, rng):
        h0_qpos = id_cache.joint_qpos_idx["h0"]
        h1_qpos = id_cache.joint_qpos_idx["h1"]

        half_dist = rng.uniform(0.75, 1.25)

        # H0 position & orientation
        data.qpos[h0_qpos[0]] = -half_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h0_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h0_qpos[2]] = 1.4
        yaw0 = rng.uniform(-0.2, 0.2)
        data.qpos[h0_qpos[3]] = np.cos(yaw0 / 2)
        data.qpos[h0_qpos[4]] = 0.0
        data.qpos[h0_qpos[5]] = 0.0
        data.qpos[h0_qpos[6]] = np.sin(yaw0 / 2)

        # H1 position & orientation
        data.qpos[h1_qpos[0]] = half_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[2]] = 1.4
        yaw1 = np.pi + rng.uniform(-0.2, 0.2)
        data.qpos[h1_qpos[3]] = np.cos(yaw1 / 2)
        data.qpos[h1_qpos[4]] = 0.0
        data.qpos[h1_qpos[5]] = 0.0
        data.qpos[h1_qpos[6]] = np.sin(yaw1 / 2)

        data.qvel[:] = 0.0

    @property
    def task_obs_dim(self) -> int:
        return 3  # l_arm_contact, r_arm_contact, partner_arm_contact

    def compute_task_obs(self, data, id_cache, agent, contact_info):
        partner = "h1" if agent == "h0" else "h0"
        l_arm = float(contact_info.get(f"{agent}_l_arm_{partner}_torso", False))
        r_arm = float(contact_info.get(f"{agent}_r_arm_{partner}_torso", False))
        partner_arm = float(contact_info.get(f"{partner}_arm_{agent}_torso", False))
        return np.array([l_arm, r_arm, partner_arm], dtype=np.float32)

    def compute_reward(self, data, id_cache, contact_info, ctrl,
                       contact_force_proxy, hold_steps, success, fallen):
        w = self._weights
        info = {}
        total = 0.0

        # Distance reward
        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        dist = np.linalg.norm(h0_chest - h1_chest)
        r_dist = w["w_dist"] * np.exp(-ALPHA_DIST * dist)
        total += r_dist
        info["chest_distance"] = dist
        info["r_distance"] = r_dist

        # Facing reward
        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        r_face = w["w_face"] * max(0.0, facing)
        total += r_face
        info["facing_alignment"] = facing
        info["r_facing"] = r_face

        # Stability reward
        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        r_stab = w["w_stab"] * np.exp(-BETA_SPEED * rel_speed)
        total += r_stab
        info["relative_speed"] = rel_speed
        info["r_stability"] = r_stab

        # Contact reward
        n_contacts = int(contact_info.get("h0_arm_h1_torso", False)) + \
                     int(contact_info.get("h1_arm_h0_torso", False))
        r_contact = w["w_contact"] * n_contacts
        total += r_contact
        info["r_contact"] = r_contact

        # Hand-to-back shaping
        if w["w_hand"] > 0:
            alpha = 4.0
            h0l = id_cache.get_site_xpos(data, "h0_lhand")
            h0r = id_cache.get_site_xpos(data, "h0_rhand")
            h1l = id_cache.get_site_xpos(data, "h1_lhand")
            h1r = id_cache.get_site_xpos(data, "h1_rhand")
            h0b = id_cache.get_site_xpos(data, "h0_back")
            h1b = id_cache.get_site_xpos(data, "h1_back")
            hand_r = (np.exp(-alpha * np.linalg.norm(h0l - h1b)) +
                      np.exp(-alpha * np.linalg.norm(h0r - h1b)) +
                      np.exp(-alpha * np.linalg.norm(h1l - h0b)) +
                      np.exp(-alpha * np.linalg.norm(h1r - h0b)))
            r_hand = w["w_hand"] * hand_r
            total += r_hand
            info["r_hand_to_back"] = r_hand
        else:
            info["r_hand_to_back"] = 0.0

        # Penalties
        r_energy = w["w_energy"] * np.sum(np.square(ctrl))
        total += r_energy
        info["r_energy"] = r_energy

        r_impact = w["w_impact"] * contact_force_proxy
        total += r_impact
        info["r_impact"] = r_impact

        if fallen:
            total += w["w_fall"]
            info["r_fall"] = w["w_fall"]
        else:
            info["r_fall"] = 0.0

        if success:
            total += w["r_success"]
            info["r_success"] = w["r_success"]
        else:
            info["r_success"] = 0.0

        info["total_reward"] = total
        info["hold_steps"] = hold_steps
        return total, info

    def check_success(self, data, id_cache, contact_info):
        info = {}

        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        dist = np.linalg.norm(h0_chest - h1_chest)
        dist_ok = D_MIN < dist < D_MAX
        info["hug_dist"] = dist
        info["hug_dist_ok"] = dist_ok

        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        facing_ok = facing > FACING_THRESH
        info["hug_facing"] = facing
        info["hug_facing_ok"] = facing_ok

        contact_ok = (contact_info.get("h0_arm_h1_torso", False) and
                      contact_info.get("h1_arm_h0_torso", False))
        info["hug_contact_ok"] = contact_ok

        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        speed_ok = rel_speed < V_THRESH
        info["hug_rel_speed"] = rel_speed
        info["hug_speed_ok"] = speed_ok

        h0_tilt = compute_tilt_angle(get_up_vector(h0_xmat))
        h1_tilt = compute_tilt_angle(get_up_vector(h1_xmat))
        upright_ok = h0_tilt < TILT_THRESH and h1_tilt < TILT_THRESH
        info["hug_upright_ok"] = upright_ok

        condition = dist_ok and facing_ok and contact_ok and speed_ok and upright_ok
        info["hug_condition"] = condition
        return condition, info

    def set_stage(self, stage: int) -> None:
        self._stage = stage
        self._weights = _HUG_STAGES.get(stage, _HUG_STAGES[0]).copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self._weights.copy()
