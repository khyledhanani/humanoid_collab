"""Hug task: Two agents approach, embrace, and hold a stable hug."""

from typing import Dict, List, Tuple
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


# Curriculum weights per stage.
# Hug curriculum for fixed-standing:
#   stage 0: hand placement and pose organization
#   stage 1: add initial contact requirement
#   stage 2: require bilateral contact quality + hand-on-back contact bonus
#   stage 3: tighten hold quality / wrap quality + stronger hand-on-back bonus
_HUG_STAGES = {
    0: dict(
        w_dist=0.45,
        w_face=0.08,
        w_stab=0.05,
        w_contact=1.0,
        w_hand=1.3,
        w_hand_back_contact=0.0,
        w_time=-0.02,
        w_energy=-0.001,
        w_impact=-0.005,
        w_fall=-25.0,
        r_success=300.0,
    ),
    1: dict(
        w_dist=0.4,
        w_face=0.08,
        w_stab=0.08,
        w_contact=2.0,
        w_hand=1.2,
        w_hand_back_contact=0.0,
        w_time=-0.02,
        w_energy=-0.001,
        w_impact=-0.005,
        w_fall=-25.0,
        r_success=320.0,
    ),
    2: dict(
        w_dist=0.35,
        w_face=0.07,
        w_stab=0.1,
        w_contact=3.2,
        w_hand=1.0,
        w_hand_back_contact=1.0,
        w_time=-0.02,
        w_energy=-0.001,
        w_impact=-0.007,
        w_fall=-25.0,
        r_success=8000.0,
    ),
    3: dict(
        w_dist=0.25,
        w_face=0.06,
        w_stab=0.12,
        w_contact=4.2,
        w_hand=0.7,
        w_hand_back_contact=1.4,
        w_time=-0.02,
        w_energy=-0.001,
        w_impact=-0.01,
        w_fall=-25.0,
        r_success=1000.0,
    ),
}

# Hug condition thresholds
FACING_THRESH_BASE = 0.45
FACING_THRESH_FINAL = 0.60
V_THRESH_BASE = 1.20
V_THRESH_FINAL = 0.90
TILT_THRESH = 0.5
HAND_ALPHA = 4.0
BETA_SPEED = 2.0
HAND_BACKSIDE_X_MAX = -0.01


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
        self._fixed_standing = False
        self._control_mode = "all"

    def get_contact_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("h0_arm_h1_torso", "h0_arm", "h1_torso"),
            ("h1_arm_h0_torso", "h1_arm", "h0_torso"),
            ("h0_l_arm_h1_torso", "h0_l_arm", "h1_torso"),
            ("h0_r_arm_h1_torso", "h0_r_arm", "h1_torso"),
            ("h1_l_arm_h0_torso", "h1_l_arm", "h0_torso"),
            ("h1_r_arm_h0_torso", "h1_r_arm", "h0_torso"),
            ("h0_hand_h1_torso", "h0_hand", "h1_torso"),
            ("h1_hand_h0_torso", "h1_hand", "h0_torso"),
            ("h0_l_hand_h1_torso", "h0_l_hand", "h1_torso"),
            ("h0_r_hand_h1_torso", "h0_r_hand", "h1_torso"),
            ("h1_l_hand_h0_torso", "h1_l_hand", "h0_torso"),
            ("h1_r_hand_h0_torso", "h1_r_hand", "h0_torso"),
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
        # [chest_delta_local(3), partner_back_delta_local(3),
        #  own_l/r_hand_to_partner_back(2), partner_l/r_hand_to_own_back(2),
        #  chest_dist, facing, rel_speed, directional_contact_count,
        #  side_contact_count_norm, wrap_quality]
        return 16

    def compute_task_obs(self, data, id_cache, agent, contact_info):
        if agent == "h0":
            own = "h0"
            partner = "h1"
        elif agent == "h1":
            own = "h1"
            partner = "h0"
        else:
            raise ValueError(f"Unknown agent '{agent}'")

        own_chest = id_cache.get_site_xpos(data, f"{own}_chest")
        partner_chest = id_cache.get_site_xpos(data, f"{partner}_chest")
        own_back = id_cache.get_site_xpos(data, f"{own}_back")
        partner_back = id_cache.get_site_xpos(data, f"{partner}_back")
        own_lhand = id_cache.get_site_xpos(data, f"{own}_lhand")
        own_rhand = id_cache.get_site_xpos(data, f"{own}_rhand")
        partner_lhand = id_cache.get_site_xpos(data, f"{partner}_lhand")
        partner_rhand = id_cache.get_site_xpos(data, f"{partner}_rhand")

        own_torso_xmat = id_cache.get_torso_xmat(data, own)

        chest_delta_world = partner_chest - own_chest
        partner_back_delta_world = partner_back - own_chest
        chest_delta_local = own_torso_xmat.T @ chest_delta_world
        partner_back_delta_local = own_torso_xmat.T @ partner_back_delta_world

        chest_dist = float(np.linalg.norm(chest_delta_world))

        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = float(
            compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        )

        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        rel_speed = float(np.linalg.norm(h0_vel - h1_vel))

        own_l_to_partner_back = float(np.linalg.norm(own_lhand - partner_back))
        own_r_to_partner_back = float(np.linalg.norm(own_rhand - partner_back))
        partner_l_to_own_back = float(np.linalg.norm(partner_lhand - own_back))
        partner_r_to_own_back = float(np.linalg.norm(partner_rhand - own_back))

        side_contact_count = int(contact_info.get("h0_l_arm_h1_torso", False))
        side_contact_count += int(contact_info.get("h0_r_arm_h1_torso", False))
        side_contact_count += int(contact_info.get("h1_l_arm_h0_torso", False))
        side_contact_count += int(contact_info.get("h1_r_arm_h0_torso", False))
        directional_contact_count = int(contact_info.get("h0_arm_h1_torso", False))
        directional_contact_count += int(contact_info.get("h1_arm_h0_torso", False))
        wrap_quality = 1.0 if side_contact_count >= 3 else 0.0

        return np.asarray(
            [
                chest_delta_local[0],
                chest_delta_local[1],
                chest_delta_local[2],
                partner_back_delta_local[0],
                partner_back_delta_local[1],
                partner_back_delta_local[2],
                own_l_to_partner_back,
                own_r_to_partner_back,
                partner_l_to_own_back,
                partner_r_to_own_back,
                chest_dist,
                facing,
                rel_speed,
                float(directional_contact_count),
                float(side_contact_count) / 4.0,
                wrap_quality,
            ],
            dtype=np.float32,
        )

    def _get_success_thresholds(self) -> Dict[str, object]:
        # Relaxed chest-distance windows for fixed-standing (roots are pinned).
        if self._fixed_standing:
            by_stage = {
                0: dict(dist_min=0.12, dist_max=0.90, hand_mean_max=0.42, hand_max_max=0.46, need_contact=False, need_dual_contact=False, need_wrap=False),
                1: dict(dist_min=0.12, dist_max=0.86, hand_mean_max=0.50, hand_max_max=0.46, need_contact=True, need_dual_contact=False, need_wrap=False),
                2: dict(dist_min=0.12, dist_max=0.80, hand_mean_max=0.48, hand_max_max=0.42, need_contact=True, need_dual_contact=True, need_wrap=False),
                3: dict(dist_min=0.12, dist_max=0.74, hand_mean_max=0.42, hand_max_max=0.38, need_contact=True, need_dual_contact=True, need_wrap=True),
            }
        else:
            by_stage = {
                0: dict(dist_min=0.25, dist_max=0.95, hand_mean_max=0.42, hand_max_max=0.46, need_contact=False, need_dual_contact=False, need_wrap=False),
                1: dict(dist_min=0.25, dist_max=0.80, hand_mean_max=0.50, hand_max_max=0.46, need_contact=True, need_dual_contact=False, need_wrap=False),
                2: dict(dist_min=0.25, dist_max=0.70, hand_mean_max=0.42, hand_max_max=0.42, need_contact=True, need_dual_contact=True, need_wrap=False),
                3: dict(dist_min=0.25, dist_max=0.62, hand_mean_max=0.36, hand_max_max=0.38, need_contact=True, need_dual_contact=True, need_wrap=True),
            }
        return by_stage.get(self._stage, by_stage[0])

    def _get_dist_target(self) -> float:
        if self._fixed_standing:
            target_by_stage = {0: 0.26, 1: 0.24, 2: 0.22, 3: 0.20}
        else:
            target_by_stage = {0: 0.70, 1: 0.60, 2: 0.52, 3: 0.46}
        return target_by_stage.get(self._stage, target_by_stage[0])

    def _get_facing_speed_thresholds(self) -> Tuple[float, float]:
        # Interpolate thresholds across 4 stages.
        t = float(np.clip(self._stage / 3.0, 0.0, 1.0))
        facing_thresh = (1.0 - t) * FACING_THRESH_BASE + t * FACING_THRESH_FINAL
        speed_thresh = (1.0 - t) * V_THRESH_BASE + t * V_THRESH_FINAL
        return facing_thresh, speed_thresh

    @staticmethod
    def _compute_contact_quality(contact_info: Dict[str, bool]) -> Dict[str, float]:
        side_contact_count = int(contact_info.get("h0_l_arm_h1_torso", False))
        side_contact_count += int(contact_info.get("h0_r_arm_h1_torso", False))
        side_contact_count += int(contact_info.get("h1_l_arm_h0_torso", False))
        side_contact_count += int(contact_info.get("h1_r_arm_h0_torso", False))

        h0_directional = bool(contact_info.get("h0_arm_h1_torso", False))
        h1_directional = bool(contact_info.get("h1_arm_h0_torso", False))
        directional_contact_count = int(h0_directional) + int(h1_directional)
        dual_contact = h0_directional and h1_directional

        wrap_quality = side_contact_count >= 3

        return {
            "side_contact_count": float(side_contact_count),
            "directional_contact_count": float(directional_contact_count),
            "contact_any": 1.0 if side_contact_count > 0 else 0.0,
            "dual_contact": 1.0 if dual_contact else 0.0,
            "wrap_quality": 1.0 if wrap_quality else 0.0,
        }

    @staticmethod
    def _compute_hand_back_distances(data, id_cache) -> Dict[str, float]:
        h0l = id_cache.get_site_xpos(data, "h0_lhand")
        h0r = id_cache.get_site_xpos(data, "h0_rhand")
        h1l = id_cache.get_site_xpos(data, "h1_lhand")
        h1r = id_cache.get_site_xpos(data, "h1_rhand")
        h0b = id_cache.get_site_xpos(data, "h0_back")
        h1b = id_cache.get_site_xpos(data, "h1_back")

        d_h0l_h1b = float(np.linalg.norm(h0l - h1b))
        d_h0r_h1b = float(np.linalg.norm(h0r - h1b))
        d_h1l_h0b = float(np.linalg.norm(h1l - h0b))
        d_h1r_h0b = float(np.linalg.norm(h1r - h0b))

        hand_mean = 0.25 * (d_h0l_h1b + d_h0r_h1b + d_h1l_h0b + d_h1r_h0b)
        hand_max = max(d_h0l_h1b, d_h0r_h1b, d_h1l_h0b, d_h1r_h0b)

        return {
            "d_h0l_h1b": d_h0l_h1b,
            "d_h0r_h1b": d_h0r_h1b,
            "d_h1l_h0b": d_h1l_h0b,
            "d_h1r_h0b": d_h1r_h0b,
            "hand_back_mean": hand_mean,
            "hand_back_max": hand_max,
        }

    def _get_hand_back_contact_dist_thresh(self) -> float:
        """Distance threshold used to qualify hand-on-back contact quality."""
        if self._fixed_standing:
            by_stage = {0: 0.34, 1: 0.34, 2: 0.32, 3: 0.30}
        else:
            by_stage = {0: 0.32, 1: 0.32, 2: 0.30, 3: 0.28}
        return by_stage.get(self._stage, by_stage[0])

    @staticmethod
    def _hand_local_x_wrt_torso(hand_world: np.ndarray, torso_pos: np.ndarray, torso_xmat: np.ndarray) -> float:
        """Return hand x-position in torso local frame; negative x is behind torso."""
        hand_local = torso_xmat.T @ (hand_world - torso_pos)
        return float(hand_local[0])

    def compute_reward(self, data, id_cache, contact_info, ctrl,
                       contact_force_proxy, hold_steps, success, fallen):
        w = self._weights
        info = {}
        total = 0.0

        # Distance reward
        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        dist = np.linalg.norm(h0_chest - h1_chest)
        dist_target = self._get_dist_target()
        dist_sigma = 0.26 if self._fixed_standing else 0.22
        z = (dist - dist_target) / max(dist_sigma, 1e-6)
        r_dist = w["w_dist"] * np.exp(-(z * z))
        total += r_dist
        info["chest_distance"] = dist
        info["dist_target"] = dist_target
        info["r_distance"] = r_dist

        # Facing reward
        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        proximity_gate = np.clip((1.2 - dist) / 1.2, 0.0, 1.0)
        contact_proximity_gate = 1.0
        if (not self._fixed_standing) and self._stage == 0:
            # During locomotion-only stage 0, suppress contact shaping when far apart
            # and ramp it in once agents get close enough to plausibly wrap.
            contact_proximity_gate = np.clip((1.4 - dist) / 0.6, 0.0, 1.0)
        r_face = w["w_face"] * max(0.0, facing) * proximity_gate
        total += r_face
        info["facing_alignment"] = facing
        info["proximity_gate"] = proximity_gate
        info["contact_proximity_gate"] = contact_proximity_gate
        info["r_facing"] = r_face

        # Stability reward
        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        quality = self._compute_contact_quality(contact_info)
        contact_gate = 0.25 + 0.75 * quality["contact_any"]
        r_stab = w["w_stab"] * np.exp(-BETA_SPEED * rel_speed) * contact_gate
        total += r_stab
        info["relative_speed"] = rel_speed
        info["contact_gate"] = contact_gate
        info["r_stability"] = r_stab

        # Contact reward (quality-weighted, not just any binary touch).
        r_contact = w["w_contact"] * (
            0.35 * quality["side_contact_count"]
            + 0.65 * quality["directional_contact_count"]
            + 0.75 * quality["dual_contact"]
            + 0.50 * quality["wrap_quality"]
        )
        r_contact *= contact_proximity_gate
        total += r_contact
        info["side_contact_count"] = quality["side_contact_count"]
        info["directional_contact_count"] = quality["directional_contact_count"]
        info["contact_any"] = quality["contact_any"]
        info["dual_contact"] = quality["dual_contact"]
        info["wrap_quality"] = quality["wrap_quality"]
        info["r_contact"] = r_contact

        # Hand-to-back shaping
        if w["w_hand"] > 0:
            hand_d = self._compute_hand_back_distances(data, id_cache)
            hand_r = (
                np.exp(-HAND_ALPHA * hand_d["d_h0l_h1b"])
                + np.exp(-HAND_ALPHA * hand_d["d_h0r_h1b"])
                + np.exp(-HAND_ALPHA * hand_d["d_h1l_h0b"])
                + np.exp(-HAND_ALPHA * hand_d["d_h1r_h0b"])
            ) / 4.0
            r_hand = w["w_hand"] * hand_r
            r_hand *= contact_proximity_gate
            total += r_hand
            info["h0_lhand_to_h1_back"] = hand_d["d_h0l_h1b"]
            info["h0_rhand_to_h1_back"] = hand_d["d_h0r_h1b"]
            info["h1_lhand_to_h0_back"] = hand_d["d_h1l_h0b"]
            info["h1_rhand_to_h0_back"] = hand_d["d_h1r_h0b"]
            info["hand_back_mean"] = hand_d["hand_back_mean"]
            info["hand_back_max"] = hand_d["hand_back_max"]
            info["r_hand_to_back"] = r_hand
        else:
            hand_d = self._compute_hand_back_distances(data, id_cache)
            info["h0_lhand_to_h1_back"] = hand_d["d_h0l_h1b"]
            info["h0_rhand_to_h1_back"] = hand_d["d_h0r_h1b"]
            info["h1_lhand_to_h0_back"] = hand_d["d_h1l_h0b"]
            info["h1_rhand_to_h0_back"] = hand_d["d_h1r_h0b"]
            info["hand_back_mean"] = hand_d["hand_back_mean"]
            info["hand_back_max"] = hand_d["hand_back_max"]
            info["r_hand_to_back"] = 0.0

        # Extra bonus for genuine hand-on-back contact quality (stages 2+).
        hand_back_contact_thresh = self._get_hand_back_contact_dist_thresh()
        h0_torso_pos = id_cache.get_torso_xpos(data, "h0")
        h1_torso_pos = id_cache.get_torso_xpos(data, "h1")
        h0_torso_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_torso_xmat = id_cache.get_torso_xmat(data, "h1")
        h0l = id_cache.get_site_xpos(data, "h0_lhand")
        h0r = id_cache.get_site_xpos(data, "h0_rhand")
        h1l = id_cache.get_site_xpos(data, "h1_lhand")
        h1r = id_cache.get_site_xpos(data, "h1_rhand")
        h0_l_local_x_on_h1 = self._hand_local_x_wrt_torso(h0l, h1_torso_pos, h1_torso_xmat)
        h0_r_local_x_on_h1 = self._hand_local_x_wrt_torso(h0r, h1_torso_pos, h1_torso_xmat)
        h1_l_local_x_on_h0 = self._hand_local_x_wrt_torso(h1l, h0_torso_pos, h0_torso_xmat)
        h1_r_local_x_on_h0 = self._hand_local_x_wrt_torso(h1r, h0_torso_pos, h0_torso_xmat)
        h0_l_on_back = (
            bool(contact_info.get("h0_l_hand_h1_torso", False))
            and (hand_d["d_h0l_h1b"] < hand_back_contact_thresh)
            and (h0_l_local_x_on_h1 < HAND_BACKSIDE_X_MAX)
        )
        h0_r_on_back = (
            bool(contact_info.get("h0_r_hand_h1_torso", False))
            and (hand_d["d_h0r_h1b"] < hand_back_contact_thresh)
            and (h0_r_local_x_on_h1 < HAND_BACKSIDE_X_MAX)
        )
        h1_l_on_back = (
            bool(contact_info.get("h1_l_hand_h0_torso", False))
            and (hand_d["d_h1l_h0b"] < hand_back_contact_thresh)
            and (h1_l_local_x_on_h0 < HAND_BACKSIDE_X_MAX)
        )
        h1_r_on_back = (
            bool(contact_info.get("h1_r_hand_h0_torso", False))
            and (hand_d["d_h1r_h0b"] < hand_back_contact_thresh)
            and (h1_r_local_x_on_h0 < HAND_BACKSIDE_X_MAX)
        )
        hand_back_contact_count = float(
            int(h0_l_on_back) + int(h0_r_on_back) + int(h1_l_on_back) + int(h1_r_on_back)
        )
        hand_back_contact_ratio = hand_back_contact_count / 4.0
        w_hand_back_contact = float(w.get("w_hand_back_contact", 0.0))
        r_hand_back_contact = w_hand_back_contact * hand_back_contact_ratio
        r_hand_back_contact *= contact_proximity_gate
        total += r_hand_back_contact
        info["hand_back_contact_dist_thresh"] = hand_back_contact_thresh
        info["hand_backside_x_max"] = HAND_BACKSIDE_X_MAX
        info["h0_lhand_local_x_on_h1"] = h0_l_local_x_on_h1
        info["h0_rhand_local_x_on_h1"] = h0_r_local_x_on_h1
        info["h1_lhand_local_x_on_h0"] = h1_l_local_x_on_h0
        info["h1_rhand_local_x_on_h0"] = h1_r_local_x_on_h0
        info["hand_back_contact_count"] = hand_back_contact_count
        info["hand_back_contact_ratio"] = hand_back_contact_ratio
        info["r_hand_back_contact"] = r_hand_back_contact

        # Time pressure to prefer completing the task quickly over farming shaping.
        r_time = float(w.get("w_time", 0.0))
        total += r_time
        info["r_time"] = r_time

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
        thresholds = self._get_success_thresholds()

        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        dist = np.linalg.norm(h0_chest - h1_chest)
        dist_ok = thresholds["dist_min"] < dist < thresholds["dist_max"]
        info["hug_dist"] = dist
        info["hug_dist_min"] = thresholds["dist_min"]
        info["hug_dist_max"] = thresholds["dist_max"]
        info["hug_dist_ok"] = dist_ok

        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        facing_thresh, speed_thresh = self._get_facing_speed_thresholds()
        facing_ok = facing > facing_thresh
        info["hug_facing"] = facing
        info["hug_facing_thresh"] = facing_thresh
        info["hug_facing_ok"] = facing_ok

        quality = self._compute_contact_quality(contact_info)
        contact_any = bool(quality["contact_any"])
        dual_contact = bool(quality["dual_contact"])
        wrap_ok = bool(quality["wrap_quality"])
        info["hug_contact_any"] = contact_any
        info["hug_dual_contact"] = dual_contact
        info["hug_wrap_ok"] = wrap_ok

        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        speed_ok = rel_speed < speed_thresh
        info["hug_rel_speed"] = rel_speed
        info["hug_speed_thresh"] = speed_thresh
        info["hug_speed_ok"] = speed_ok

        h0_tilt = compute_tilt_angle(get_up_vector(h0_xmat))
        h1_tilt = compute_tilt_angle(get_up_vector(h1_xmat))
        upright_ok = h0_tilt < TILT_THRESH and h1_tilt < TILT_THRESH
        info["hug_upright_ok"] = upright_ok

        hand_d = self._compute_hand_back_distances(data, id_cache)
        hand_mean_ok = hand_d["hand_back_mean"] < thresholds["hand_mean_max"]
        hand_max_ok = hand_d["hand_back_max"] < thresholds["hand_max_max"]
        info["hug_hand_back_mean"] = hand_d["hand_back_mean"]
        info["hug_hand_back_maxdist"] = hand_d["hand_back_max"]
        info["hug_hand_back_max"] = thresholds["hand_mean_max"]
        info["hug_hand_back_max_allow"] = thresholds["hand_max_max"]
        info["hug_hand_back_ok"] = hand_mean_ok
        info["hug_hand_back_max_ok"] = hand_max_ok

        condition = dist_ok and facing_ok and speed_ok and upright_ok and hand_mean_ok and hand_max_ok
        if thresholds["need_contact"]:
            condition = condition and contact_any
        if thresholds["need_dual_contact"]:
            condition = condition and dual_contact
        if thresholds["need_wrap"]:
            condition = condition and wrap_ok

        info["hug_need_contact"] = bool(thresholds["need_contact"])
        info["hug_need_dual_contact"] = bool(thresholds["need_dual_contact"])
        info["hug_need_wrap"] = bool(thresholds["need_wrap"])
        info["hug_condition"] = condition
        return condition, info

    def set_stage(self, stage: int) -> None:
        self._stage = stage
        self._weights = _HUG_STAGES.get(stage, _HUG_STAGES[0]).copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self._weights.copy()
