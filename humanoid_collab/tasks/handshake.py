"""Handshake task: Two agents approach, extend hands, and shake."""

from typing import Dict, List, Tuple, Any
import numpy as np
import mujoco

from humanoid_collab.tasks.base import TaskConfig
from humanoid_collab.tasks.registry import register_task
from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    get_forward_vector,
    compute_facing_alignment,
    get_root_linear_velocity,
)
from humanoid_collab.constants import DEFAULT_ROOT_Z


# Simplified curriculum - focus on hand proximity and contact
_HANDSHAKE_STAGES = {
    # Stage 0: Very strong hand proximity shaping for initial learning
    0: dict(
        w_hand_dist=3.0,      # Very strong shaping to learn arm control
        w_contact=10.0,       # Bonus per step when in contact
        w_approach=0.5,       # Locomotion: reward approaching (non-fixed only)
        w_fall=-10.0,
        r_success=100.0,
    ),
    # Stage 1: Strong shaping + contact
    1: dict(
        w_hand_dist=2.5,
        w_contact=15.0,
        w_approach=0.3,
        w_fall=-10.0,
        r_success=100.0,
    ),
    # Stage 2: Moderate shaping, strong contact
    2: dict(
        w_hand_dist=2.0,
        w_contact=20.0,
        w_approach=0.1,
        w_fall=-10.0,
        r_success=150.0,
    ),
    # Stage 3: Lighter shaping, very strong contact (for fine-tuning)
    3: dict(
        w_hand_dist=1.0,
        w_contact=25.0,
        w_approach=0.0,
        w_fall=-10.0,
        r_success=200.0,
    ),
}


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
        self._fixed_standing = False
        self._control_mode = "all"

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
            ("h0_hand_h1_hand", "h0_hand", "h1_hand"),
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
        data.qpos[h0_qpos[2]] = DEFAULT_ROOT_Z
        yaw0 = rng.uniform(-0.2, 0.2)
        data.qpos[h0_qpos[3]] = np.cos(yaw0 / 2)
        data.qpos[h0_qpos[4]] = 0.0
        data.qpos[h0_qpos[5]] = 0.0
        data.qpos[h0_qpos[6]] = np.sin(yaw0 / 2)

        # H1 position & orientation
        data.qpos[h1_qpos[0]] = half_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[2]] = DEFAULT_ROOT_Z
        yaw1 = np.pi + rng.uniform(-0.2, 0.2)
        data.qpos[h1_qpos[3]] = np.cos(yaw1 / 2)
        data.qpos[h1_qpos[4]] = 0.0
        data.qpos[h1_qpos[5]] = 0.0
        data.qpos[h1_qpos[6]] = np.sin(yaw1 / 2)

        data.qvel[:] = 0.0

    @property
    def task_obs_dim(self) -> int:
        return 12

    def compute_task_obs(self, data, id_cache, agent, contact_info):
        if agent == "h0":
            own_rhand_site = "h0_rhand"
            partner_rhand_site = "h1_rhand"
            own_chest_site = "h0_chest"
            partner_chest_site = "h1_chest"
        elif agent == "h1":
            own_rhand_site = "h1_rhand"
            partner_rhand_site = "h0_rhand"
            own_chest_site = "h1_chest"
            partner_chest_site = "h0_chest"
        else:
            raise ValueError(f"Unknown agent '{agent}'")

        own_rhand = id_cache.get_site_xpos(data, own_rhand_site)
        partner_rhand = id_cache.get_site_xpos(data, partner_rhand_site)
        own_chest = id_cache.get_site_xpos(data, own_chest_site)
        partner_chest = id_cache.get_site_xpos(data, partner_chest_site)

        hand_delta_world = partner_rhand - own_rhand
        chest_delta_world = partner_chest - own_chest

        torso_xmat = id_cache.get_torso_xmat(data, agent)
        hand_delta_local = torso_xmat.T @ hand_delta_world
        chest_delta_local = torso_xmat.T @ chest_delta_world

        hand_dist = float(np.linalg.norm(hand_delta_world))
        chest_dist = float(np.linalg.norm(chest_delta_world))

        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        facing = float(
            compute_facing_alignment(get_forward_vector(h0_xmat), get_forward_vector(h1_xmat))
        )

        correct_contact = 1.0 if contact_info.get(self.SHAKE_PAIR_KEY, False) else 0.0
        wrong_contact = 1.0 if any(
            contact_info.get(k, False) for k in self.ALL_HAND_PAIRS if k != self.SHAKE_PAIR_KEY
        ) else 0.0

        h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
        h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])
        rel_speed = float(np.linalg.norm(h0_vel - h1_vel))

        return np.asarray(
            [
                hand_delta_local[0],
                hand_delta_local[1],
                hand_delta_local[2],
                chest_delta_local[0],
                chest_delta_local[1],
                chest_delta_local[2],
                hand_dist,
                chest_dist,
                facing,
                rel_speed,
                correct_contact,
                wrong_contact,
            ],
            dtype=np.float32,
        )

    def compute_reward(self, data, id_cache, contact_info, ctrl,
                       contact_force_proxy, hold_steps, success, fallen):
        """Simplified reward: hand distance + contact bonus + success bonus.

        Adds locomotion shaping when not in fixed_standing mode.
        """
        w = self._weights
        info = {}
        total = 0.0

        # --- Core signal: hand distance ---
        h0_rhand = id_cache.get_site_xpos(data, self.SHAKE_SITE_A)
        h1_rhand = id_cache.get_site_xpos(data, self.SHAKE_SITE_B)
        hand_dist = float(np.linalg.norm(h0_rhand - h1_rhand))
        info["hand_distance"] = hand_dist

        # Combined linear + exponential reward for hand proximity
        # Linear part: constant gradient for general approach
        # Exponential part: stronger gradient when close (encourages final approach)
        max_dist = 1.5
        linear_part = max(0.0, 1.0 - hand_dist / max_dist)
        exp_part = np.exp(-2.0 * hand_dist)  # Strong signal when < 0.5m
        r_hand_dist = w["w_hand_dist"] * (0.5 * linear_part + 0.5 * exp_part)
        total += r_hand_dist
        info["r_hand_dist"] = r_hand_dist

        # --- Locomotion shaping (only when not fixed_standing) ---
        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        chest_dist = float(np.linalg.norm(h0_chest - h1_chest))
        info["chest_distance"] = chest_dist

        r_approach = 0.0
        if not self._fixed_standing and w.get("w_approach", 0) > 0:
            # Reward agents for moving toward each other
            h0_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h0"])
            h1_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx["h1"])

            chest_delta = h1_chest - h0_chest
            chest_delta[2] = 0.0  # XY plane only
            dist_xy = np.linalg.norm(chest_delta)

            if dist_xy > 0.5:  # Only reward approach when far apart
                dir_h0_to_h1 = chest_delta / max(dist_xy, 1e-6)

                # Project velocities onto approach direction
                approach_h0 = max(0.0, float(np.dot(h0_vel[:2], dir_h0_to_h1[:2])))
                approach_h1 = max(0.0, float(np.dot(h1_vel[:2], -dir_h0_to_h1[:2])))

                # Reward is sum of approach speeds (capped)
                r_approach = w["w_approach"] * (
                    min(approach_h0, 1.0) + min(approach_h1, 1.0)
                )
        total += r_approach
        info["r_approach"] = r_approach

        # --- Contact reward: any right-to-right contact is good ---
        correct_contact = contact_info.get(self.SHAKE_PAIR_KEY, False)
        if correct_contact:
            r_contact = w["w_contact"]
        else:
            r_contact = 0.0
        total += r_contact
        info["r_contact"] = r_contact
        info["correct_hand_contact"] = correct_contact

        # --- Fall penalty ---
        if fallen:
            total += w["w_fall"]
            info["r_fall"] = w["w_fall"]
        else:
            info["r_fall"] = 0.0

        # --- Success bonus ---
        if success:
            total += w["r_success"]
            info["r_success"] = w["r_success"]
        else:
            info["r_success"] = 0.0

        info["hold_steps"] = hold_steps
        info["total_reward"] = total

        return total, info

    def check_success(self, data, id_cache, contact_info):
        """Simplified success: just need right-to-right hand contact."""
        info = {}

        # Core requirement: right-to-right hand contact
        correct_contact = contact_info.get(self.SHAKE_PAIR_KEY, False)
        info["shake_correct_contact"] = correct_contact
        info["shake_condition"] = correct_contact

        # Log extra info for debugging (not used in condition)
        h0_rhand = id_cache.get_site_xpos(data, self.SHAKE_SITE_A)
        h1_rhand = id_cache.get_site_xpos(data, self.SHAKE_SITE_B)
        info["hand_distance"] = float(np.linalg.norm(h0_rhand - h1_rhand))

        h0_chest = id_cache.get_site_xpos(data, "h0_chest")
        h1_chest = id_cache.get_site_xpos(data, "h1_chest")
        info["chest_distance"] = float(np.linalg.norm(h0_chest - h1_chest))

        return correct_contact, info

    def set_stage(self, stage: int) -> None:
        self._stage = stage
        self._weights = _HANDSHAKE_STAGES.get(stage, _HANDSHAKE_STAGES[0]).copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self._weights.copy()
