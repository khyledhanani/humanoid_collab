"""Handshake task: Two agents approach, extend hands, and shake."""

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


# Curriculum weights per stage.
_HANDSHAKE_STAGES = {
    # Stage 0: locomotion-first curriculum. No hand shaping/contact requirement.
    0: dict(
        w_dist=0.12, w_face=0.08, w_stab=0.03, w_hand_prox=0.0, w_contact=0.0,
        w_approach=0.35, w_stop=0.25, w_upright=0.15,
        w_energy=-0.0001, w_impact=-0.001, w_fall=-10.0, r_success=10.0,
    ),
    # Stage 1: transition into handshake objective.
    1: dict(
        w_dist=0.20, w_face=0.10, w_stab=0.05, w_hand_prox=0.0, w_contact=0.0,
        w_approach=0.06, w_stop=0.05, w_upright=0.04,
        w_energy=-0.0001, w_impact=-0.001, w_fall=-10.0, r_success=50.0,
    ),
    # Stage 2: emphasize right-hand proximity/contact.
    2: dict(
        w_dist=0.15, w_face=0.10, w_stab=0.05, w_hand_prox=0.30, w_contact=2.0,
        w_approach=0.05, w_stop=0.04, w_upright=0.03,
        w_energy=-0.0001, w_impact=-0.001, w_fall=-10.0, r_success=50.0,
    ),
    # Stage 3: strongest contact objective while retaining small locomotion regularization.
    3: dict(
        w_dist=0.10, w_face=0.08, w_stab=0.05, w_hand_prox=0.15, w_contact=5.0,
        w_approach=0.04, w_stop=0.03, w_upright=0.03,
        w_energy=-0.0001, w_impact=-0.001, w_fall=-10.0, r_success=100.0,
    ),
}

# Handshake thresholds
D_MIN, D_MAX = 0.4, 1.0  # Arm's length apart
FACING_THRESH = 0.6
V_THRESH = 0.8
TILT_THRESH = 0.5
ALPHA_DIST = 3.0
BETA_SPEED = 2.0
APPROACH_DIST_GATE = 0.9
STOP_ZONE_MIN = 0.6
STOP_ZONE_MAX = 0.9
LOCO_FACING_THRESH = 0.4
LOCO_SPEED_THRESH = 1.1
RESET_ROOT_Z = 1.0


@register_task
class HandshakeTask(TaskConfig):

    @property
    def name(self) -> str:
        return "handshake"

    @property
    def num_curriculum_stages(self) -> int:
        return 4

    def __init__(self):
        self._weights = _HANDSHAKE_STAGES[0].copy()
        self._stage = 0

    # Right-to-right handshake (the human convention)
    SHAKE_PAIR_KEY = "h0_r_hand_h1_r_hand"
    SHAKE_SITE_A = "h0_rhand"
    SHAKE_SITE_B = "h1_rhand"

    # All hand pairs we monitor (to detect and penalize wrong-hand contacts)
    ALL_HAND_PAIRS = [
        "h0_r_hand_h1_r_hand",
        "h0_r_hand_h1_l_hand",
        "h0_l_hand_h1_r_hand",
        "h0_l_hand_h1_l_hand",
    ]

    def get_contact_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("h0_r_hand_h1_r_hand", "h0_r_hand", "h1_r_hand"),
            ("h0_r_hand_h1_l_hand", "h0_r_hand", "h1_l_hand"),
            ("h0_l_hand_h1_r_hand", "h0_l_hand", "h1_r_hand"),
            ("h0_l_hand_h1_l_hand", "h0_l_hand", "h1_l_hand"),
        ]

    def randomize_state(self, model, data, id_cache, rng):
        h0_qpos = id_cache.joint_qpos_idx["h0"]
        h1_qpos = id_cache.joint_qpos_idx["h1"]

        half_dist = rng.uniform(0.75, 1.25)

        # H0 position & orientation
        data.qpos[h0_qpos[0]] = -half_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h0_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h0_qpos[2]] = RESET_ROOT_Z
        yaw0 = rng.uniform(-0.2, 0.2)
        data.qpos[h0_qpos[3]] = np.cos(yaw0 / 2)
        data.qpos[h0_qpos[4]] = 0.0
        data.qpos[h0_qpos[5]] = 0.0
        data.qpos[h0_qpos[6]] = np.sin(yaw0 / 2)

        # H1 position & orientation
        data.qpos[h1_qpos[0]] = half_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[2]] = RESET_ROOT_Z
        yaw1 = np.pi + rng.uniform(-0.2, 0.2)
        data.qpos[h1_qpos[3]] = np.cos(yaw1 / 2)
        data.qpos[h1_qpos[4]] = 0.0
        data.qpos[h1_qpos[5]] = 0.0
        data.qpos[h1_qpos[6]] = np.sin(yaw1 / 2)

        data.qvel[:] = 0.0

    @property
    def task_obs_dim(self) -> int:
        return 0

    def compute_task_obs(self, data, id_cache, agent, contact_info):
        return np.zeros((0,), dtype=np.float32)

    def compute_reward(self, data, id_cache, contact_info, ctrl,
                       contact_force_proxy, hold_steps, success, fallen):
        w = self._weights
        info = {}
        total = 0.0

        # Distance reward (target ~0.7m, not too close)
        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        dist = np.linalg.norm(h0_chest - h1_chest)
        # Reward being close, but optimal around 0.7m
        target_dist = 0.7
        dist_error = abs(dist - target_dist)
        r_dist = w["w_dist"] * np.exp(-ALPHA_DIST * dist_error)
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

        # Locomotion shaping: encourage approach at long range.
        chest_delta = h1_chest - h0_chest
        chest_delta[2] = 0.0
        chest_dist_xy = np.linalg.norm(chest_delta)
        if chest_dist_xy > 1e-6:
            dir_h0_to_h1 = chest_delta / chest_dist_xy
        else:
            dir_h0_to_h1 = np.zeros_like(chest_delta)
        dir_h1_to_h0 = -dir_h0_to_h1

        h0_v_xy = h0_vel.copy()
        h1_v_xy = h1_vel.copy()
        h0_v_xy[2] = 0.0
        h1_v_xy[2] = 0.0

        approach_speed_h0 = max(0.0, float(np.dot(h0_v_xy, dir_h0_to_h1)))
        approach_speed_h1 = max(0.0, float(np.dot(h1_v_xy, dir_h1_to_h0)))
        approach_sum = np.clip(approach_speed_h0, 0.0, 1.5) + np.clip(approach_speed_h1, 0.0, 1.5)
        if dist > APPROACH_DIST_GATE:
            r_approach = w["w_approach"] * approach_sum
        else:
            r_approach = 0.0
        total += r_approach
        info["r_approach"] = r_approach
        info["approach_speed_h0"] = approach_speed_h0
        info["approach_speed_h1"] = approach_speed_h1

        # Locomotion shaping: encourage stopping in pre-contact zone.
        h0_speed = np.linalg.norm(h0_v_xy)
        h1_speed = np.linalg.norm(h1_v_xy)
        stop_score = np.exp(-2.0 * h0_speed) + np.exp(-2.0 * h1_speed)
        if STOP_ZONE_MIN < dist < STOP_ZONE_MAX:
            r_stop = w["w_stop"] * stop_score
        else:
            r_stop = 0.0
        total += r_stop
        info["r_stop"] = r_stop
        info["h0_speed_xy"] = h0_speed
        info["h1_speed_xy"] = h1_speed

        # Light upright regularization carried through all stages.
        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        h0_tilt = compute_tilt_angle(get_up_vector(h0_xmat))
        h1_tilt = compute_tilt_angle(get_up_vector(h1_xmat))
        upright_score = 0.5 * (
            max(0.0, 1.0 - h0_tilt / TILT_THRESH)
            + max(0.0, 1.0 - h1_tilt / TILT_THRESH)
        )
        r_upright = w["w_upright"] * upright_score
        total += r_upright
        info["r_upright"] = r_upright
        info["h0_tilt"] = h0_tilt
        info["h1_tilt"] = h1_tilt

        # --- Hand proximity shaping (right-to-right only) ---
        h0_rhand = id_cache.get_site_xpos(data, self.SHAKE_SITE_A)
        h1_rhand = id_cache.get_site_xpos(data, self.SHAKE_SITE_B)
        hand_dist = np.linalg.norm(h0_rhand - h1_rhand)

        if w["w_hand_prox"] > 0:
            r_hand = w["w_hand_prox"] * np.exp(-2.0 * hand_dist)

            # Anneal hand shaping once contact is achieved
            if contact_info.get(self.SHAKE_PAIR_KEY, False):
                r_hand *= 0.3
            total += r_hand
            info["r_hand_prox"] = r_hand
        else:
            info["r_hand_prox"] = 0.0
        info["hand_distance"] = hand_dist

        # --- Contact reward: right-to-right = reward, wrong pairs = penalty ---
        correct_contact = contact_info.get(self.SHAKE_PAIR_KEY, False)
        wrong_pairs = [k for k in self.ALL_HAND_PAIRS
                       if k != self.SHAKE_PAIR_KEY and contact_info.get(k, False)]

        if correct_contact and not wrong_pairs:
            # Clean right-to-right handshake
            r_contact = w["w_contact"]
        elif correct_contact and wrong_pairs:
            # Right-to-right plus extra hands -- partial credit but discourage
            r_contact = 0.3 * w["w_contact"]
        elif wrong_pairs:
            # Wrong hand pair only -- penalize
            r_contact = -0.5 * w["w_contact"]
        else:
            r_contact = 0.0
        total += r_contact
        info["r_contact"] = r_contact
        info["correct_hand_contact"] = correct_contact
        info["wrong_hand_pairs"] = len(wrong_pairs)

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
        info["shake_dist"] = dist
        info["shake_dist_ok"] = dist_ok

        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        facing_ok = facing > FACING_THRESH
        info["shake_facing"] = facing
        info["shake_facing_ok"] = facing_ok

        # Right-to-right hand contact required (no wrong-hand contacts)
        correct_contact = contact_info.get(self.SHAKE_PAIR_KEY, False)
        wrong_pairs = [k for k in self.ALL_HAND_PAIRS
                       if k != self.SHAKE_PAIR_KEY and contact_info.get(k, False)]
        contact_ok = correct_contact and not wrong_pairs
        info["shake_contact_ok"] = contact_ok
        info["shake_correct_contact"] = correct_contact
        info["shake_wrong_pairs"] = len(wrong_pairs)

        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        h0_speed = np.linalg.norm(h0_vel[0:2])
        h1_speed = np.linalg.norm(h1_vel[0:2])
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        speed_ok = rel_speed < V_THRESH
        info["shake_rel_speed"] = rel_speed
        info["shake_speed_ok"] = speed_ok
        info["shake_h0_speed_xy"] = h0_speed
        info["shake_h1_speed_xy"] = h1_speed

        h0_tilt = compute_tilt_angle(get_up_vector(h0_xmat))
        h1_tilt = compute_tilt_angle(get_up_vector(h1_xmat))
        upright_ok = h0_tilt < TILT_THRESH and h1_tilt < TILT_THRESH
        info["shake_upright_ok"] = upright_ok

        if self._stage == 0:
            # Stage-0 success is locomotion quality near handshake distance.
            facing_near_ok = facing > LOCO_FACING_THRESH
            stop_near_ok = max(h0_speed, h1_speed) < LOCO_SPEED_THRESH
            condition = dist_ok and facing_near_ok and stop_near_ok and upright_ok
            info["shake_contact_ok"] = True
            info["shake_stage0_facing_ok"] = facing_near_ok
            info["shake_stage0_stop_ok"] = stop_near_ok
        else:
            condition = dist_ok and facing_ok and contact_ok and speed_ok and upright_ok

        info["shake_condition"] = condition
        return condition, info

    def set_stage(self, stage: int) -> None:
        self._stage = stage
        self._weights = _HANDSHAKE_STAGES.get(stage, _HANDSHAKE_STAGES[0]).copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self._weights.copy()
