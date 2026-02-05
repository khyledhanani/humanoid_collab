"""Cooperative Box Lift task: Two agents approach a box, grip it, and lift together."""

from typing import Dict, List, Tuple, Any, Set
import numpy as np
import mujoco

from humanoid_collab.tasks.base import TaskConfig
from humanoid_collab.tasks.registry import register_task
from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    get_forward_vector,
    get_up_vector,
    rotate_to_local_frame,
    compute_tilt_angle,
    get_root_linear_velocity,
)


# Curriculum weights per stage
_BOX_LIFT_STAGES = {
    0: dict(w_approach=2.0, w_hand_box=0.0, w_grip=0.0, w_lift=0.0, w_hold=0.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=100.0),
    1: dict(w_approach=1.5, w_hand_box=3.0, w_grip=0.0, w_lift=0.0, w_hold=0.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=100.0),
    2: dict(w_approach=1.0, w_hand_box=1.5, w_grip=3.0, w_lift=2.0, w_hold=0.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=150.0),
    3: dict(w_approach=0.5, w_hand_box=1.0, w_grip=2.0, w_lift=5.0, w_hold=3.0,
            w_energy=-0.001, w_impact=-0.01, w_fall=-100.0, r_success=200.0),
}

# Box lift thresholds
TARGET_HEIGHT = 0.8  # Target box height for success
BOX_V_THRESH = 0.5  # Max box velocity for stable hold
TILT_THRESH = 0.5
ALPHA_APPROACH = 3.0


@register_task
class BoxLiftTask(TaskConfig):

    @property
    def name(self) -> str:
        return "box_lift"

    @property
    def num_curriculum_stages(self) -> int:
        return 4

    def __init__(self):
        self._weights = _BOX_LIFT_STAGES[0].copy()
        self._stage = 0

    def mjcf_worldbody_additions(self) -> str:
        return """
    <body name="box" pos="0 0 0.15">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.2 0.15 0.15"
            mass="5.0" rgba="0.8 0.2 0.2 1" condim="4" friction="1.5 0.5 0.1"/>
      <site name="box_center" pos="0 0 0" size="0.02" rgba="1 1 0 0.5"/>
      <site name="box_top" pos="0 0 0.15" size="0.02" rgba="1 1 0 0.5"/>
      <site name="box_left" pos="-0.2 0 0" size="0.02" rgba="1 1 0 0.5"/>
      <site name="box_right" pos="0.2 0 0" size="0.02" rgba="1 1 0 0.5"/>
    </body>"""

    def register_geom_groups(self, model, id_cache):
        """Register box geom group."""
        box_geoms = set()
        for i in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name is not None and geom_name.startswith("box"):
                box_geoms.add(i)
        id_cache.register_geom_group("box", box_geoms)

    def get_contact_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("h0_hand_box", "h0_hand", "box"),
            ("h1_hand_box", "h1_hand", "box"),
            ("h0_l_hand_box", "h0_l_hand", "box"),
            ("h0_r_hand_box", "h0_r_hand", "box"),
            ("h1_l_hand_box", "h1_l_hand", "box"),
            ("h1_r_hand_box", "h1_r_hand", "box"),
        ]

    def randomize_state(self, model, data, id_cache, rng):
        h0_qpos = id_cache.joint_qpos_idx["h0"]
        h1_qpos = id_cache.joint_qpos_idx["h1"]

        # Agents on opposite sides of the box (far enough that falling bodies don't reach it)
        agent_dist = rng.uniform(1.8, 2.2)

        # H0 on -x side
        data.qpos[h0_qpos[0]] = -agent_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h0_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h0_qpos[2]] = 1.4
        yaw0 = rng.uniform(-0.2, 0.2)  # Facing +x (toward box)
        data.qpos[h0_qpos[3]] = np.cos(yaw0 / 2)
        data.qpos[h0_qpos[4]] = 0.0
        data.qpos[h0_qpos[5]] = 0.0
        data.qpos[h0_qpos[6]] = np.sin(yaw0 / 2)

        # H1 on +x side
        data.qpos[h1_qpos[0]] = agent_dist + rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[1]] = rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[2]] = 1.4
        yaw1 = np.pi + rng.uniform(-0.2, 0.2)  # Facing -x (toward box)
        data.qpos[h1_qpos[3]] = np.cos(yaw1 / 2)
        data.qpos[h1_qpos[4]] = 0.0
        data.qpos[h1_qpos[5]] = 0.0
        data.qpos[h1_qpos[6]] = np.sin(yaw1 / 2)

        # Box position - find the box freejoint qpos indices
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name == "box_joint":
                box_qpos_start = model.jnt_qposadr[i]
                # Box at center with small position noise
                data.qpos[box_qpos_start + 0] = rng.uniform(-0.1, 0.1)
                data.qpos[box_qpos_start + 1] = rng.uniform(-0.1, 0.1)
                data.qpos[box_qpos_start + 2] = 0.151  # Slightly above floor to avoid initial bounce
                # Neutral orientation
                data.qpos[box_qpos_start + 3] = 1.0
                data.qpos[box_qpos_start + 4] = 0.0
                data.qpos[box_qpos_start + 5] = 0.0
                data.qpos[box_qpos_start + 6] = 0.0
                break

        data.qvel[:] = 0.0

    @property
    def task_obs_dim(self) -> int:
        # box_center_local(3) + box_height(1) + l_hand_box(1) + r_hand_box(1) +
        # partner_hand_box(1) + box_vel_mag(1) = 8
        return 8

    def compute_task_obs(self, data, id_cache, agent, contact_info):
        partner = "h1" if agent == "h0" else "h0"

        # Box center position in agent's local frame
        self_xpos = id_cache.get_torso_xpos(data, agent)
        self_xmat = id_cache.get_torso_xmat(data, agent)
        box_pos = id_cache.get_site_xpos(data, "box_center")
        rel_box = box_pos - self_xpos
        rel_box_local = rotate_to_local_frame(rel_box, self_xmat)

        # Box height
        box_height = box_pos[2]

        # Hand-box contacts
        l_hand_box = float(contact_info.get(f"{agent}_l_hand_box", False))
        r_hand_box = float(contact_info.get(f"{agent}_r_hand_box", False))
        partner_hand_box = float(contact_info.get(f"{partner}_hand_box", False))

        # Box velocity magnitude
        # Find box body velocity from qvel
        box_vel_mag = self._get_box_velocity_mag(data, id_cache)

        return np.array([
            rel_box_local[0], rel_box_local[1], rel_box_local[2],
            box_height,
            l_hand_box, r_hand_box,
            partner_hand_box,
            box_vel_mag,
        ], dtype=np.float32)

    def _get_box_velocity_mag(self, data, id_cache):
        """Get box velocity magnitude from its freejoint qvel."""
        model = id_cache.model
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name == "box_joint":
                qvel_start = model.jnt_dofadr[i]
                box_vel = data.qvel[qvel_start:qvel_start + 3]
                return float(np.linalg.norm(box_vel))
        return 0.0

    def _get_box_height(self, data, id_cache):
        """Get current box height."""
        return float(id_cache.get_site_xpos(data, "box_center")[2])

    def compute_reward(self, data, id_cache, contact_info, ctrl,
                       contact_force_proxy, hold_steps, success, fallen):
        w = self._weights
        info = {}
        total = 0.0

        box_pos = id_cache.get_site_xpos(data, "box_center")
        box_height = box_pos[2]
        info["box_height"] = box_height

        # Approach box reward (per agent)
        r_approach = 0.0
        for agent in ["h0", "h1"]:
            agent_pos = id_cache.get_torso_xpos(data, agent)
            dist_to_box = np.linalg.norm(agent_pos[:2] - box_pos[:2])  # XY distance
            r_approach += np.exp(-ALPHA_APPROACH * dist_to_box)
        r_approach = w["w_approach"] * r_approach
        total += r_approach
        info["r_approach"] = r_approach

        # Hand-to-box shaping
        if w["w_hand_box"] > 0:
            r_hand_box = 0.0
            alpha = 4.0
            for agent in ["h0", "h1"]:
                for hand_site in [f"{agent}_lhand", f"{agent}_rhand"]:
                    hand_pos = id_cache.get_site_xpos(data, hand_site)
                    dist_hand = np.linalg.norm(hand_pos - box_pos)
                    r_hand_box += np.exp(-alpha * dist_hand)
            r_hand_box = w["w_hand_box"] * r_hand_box
            total += r_hand_box
            info["r_hand_box"] = r_hand_box
        else:
            info["r_hand_box"] = 0.0

        # Grip contact reward
        n_grips = int(contact_info.get("h0_hand_box", False)) + \
                  int(contact_info.get("h1_hand_box", False))
        r_grip = w["w_grip"] * n_grips
        total += r_grip
        info["r_grip"] = r_grip
        info["n_grips"] = n_grips

        # Lift reward
        if w["w_lift"] > 0:
            lift_progress = min(1.0, box_height / TARGET_HEIGHT)
            r_lift = w["w_lift"] * lift_progress
            total += r_lift
            info["r_lift"] = r_lift
            info["lift_progress"] = lift_progress
        else:
            info["r_lift"] = 0.0

        # Height hold bonus (at target height)
        if w["w_hold"] > 0 and box_height > TARGET_HEIGHT * 0.9:
            box_vel = self._get_box_velocity_mag(data, id_cache)
            r_hold = w["w_hold"] * np.exp(-2.0 * box_vel)
            total += r_hold
            info["r_hold"] = r_hold
        else:
            info["r_hold"] = 0.0

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

        box_height = self._get_box_height(data, id_cache)
        height_ok = box_height > TARGET_HEIGHT
        info["box_height"] = box_height
        info["lift_height_ok"] = height_ok

        both_gripping = (contact_info.get("h0_hand_box", False) and
                         contact_info.get("h1_hand_box", False))
        info["lift_both_gripping"] = both_gripping

        box_vel = self._get_box_velocity_mag(data, id_cache)
        stable = box_vel < BOX_V_THRESH
        info["box_velocity"] = box_vel
        info["lift_stable"] = stable

        h0_xmat = id_cache.get_torso_xmat(data, "h0")
        h1_xmat = id_cache.get_torso_xmat(data, "h1")
        h0_tilt = compute_tilt_angle(get_up_vector(h0_xmat))
        h1_tilt = compute_tilt_angle(get_up_vector(h1_xmat))
        upright_ok = h0_tilt < TILT_THRESH and h1_tilt < TILT_THRESH
        info["lift_upright_ok"] = upright_ok

        condition = height_ok and both_gripping and stable and upright_ok
        info["lift_condition"] = condition
        return condition, info

    def set_stage(self, stage: int) -> None:
        self._stage = stage
        self._weights = _BOX_LIFT_STAGES.get(stage, _BOX_LIFT_STAGES[0]).copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self._weights.copy()
