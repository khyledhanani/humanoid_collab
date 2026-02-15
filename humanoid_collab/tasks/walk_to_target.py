"""Single-agent locomotion pretraining task: walk to a target and settle."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from humanoid_collab.constants import DEFAULT_ROOT_Z
from humanoid_collab.tasks.base import TaskConfig
from humanoid_collab.tasks.registry import register_task
from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.kinematics import (
    compute_tilt_angle,
    get_forward_vector,
    get_root_linear_velocity,
    get_up_vector,
)


_WALK_STAGES = {
    0: dict(
        w_progress=18.0,
        w_distance=0.8,
        w_heading=0.35,
        # Penalize moving backward in the torso local frame (prevents learning to backpedal/crab-walk).
        w_backpedal=-0.35,
        w_arrival=0.8,
        w_stop=0.25,
        w_face=0.12,
        w_upright=0.2,
        w_time=-0.01,
        w_energy=-0.0008,
        w_impact=-0.004,
        w_fall=-20.0,
        r_success=120.0,
        progress_clip=0.10,
        distance_scale=1.2,
    ),
    1: dict(
        w_progress=16.0,
        w_distance=0.7,
        w_heading=0.32,
        w_backpedal=-0.35,
        w_arrival=1.0,
        w_stop=0.45,
        w_face=0.18,
        w_upright=0.25,
        w_time=-0.01,
        w_energy=-0.0008,
        w_impact=-0.004,
        w_fall=-22.0,
        r_success=150.0,
        progress_clip=0.08,
        distance_scale=1.3,
    ),
    2: dict(
        w_progress=14.0,
        w_distance=0.6,
        w_heading=0.3,
        w_backpedal=-0.35,
        w_arrival=1.2,
        w_stop=0.75,
        w_face=0.24,
        w_upright=0.28,
        w_time=-0.01,
        w_energy=-0.0008,
        w_impact=-0.005,
        w_fall=-25.0,
        r_success=180.0,
        progress_clip=0.07,
        distance_scale=1.4,
    ),
    3: dict(
        w_progress=12.0,
        w_distance=0.5,
        w_heading=0.28,
        w_backpedal=-0.35,
        w_arrival=1.5,
        w_stop=1.0,
        w_face=0.3,
        w_upright=0.3,
        w_time=-0.01,
        w_energy=-0.0008,
        w_impact=-0.005,
        w_fall=-28.0,
        r_success=220.0,
        progress_clip=0.06,
        distance_scale=1.5,
    ),
}

_TARGET_DISTANCE_BY_STAGE = {
    0: (0.7, 1.5),
    1: (0.9, 2.0),
    2: (1.1, 2.6),
    3: (1.3, 3.2),
}

_SUCCESS_THRESHOLDS = {
    0: dict(arrival_dist=0.50, success_dist=0.45, success_speed=0.80, success_heading=0.15),
    1: dict(arrival_dist=0.36, success_dist=0.32, success_speed=0.55, success_heading=0.30),
    2: dict(arrival_dist=0.28, success_dist=0.24, success_speed=0.38, success_heading=0.45),
    3: dict(arrival_dist=0.24, success_dist=0.20, success_speed=0.28, success_heading=0.60),
}

_UPRIGHT_TILT_THRESH = 0.55
_TARGET_Z = 0.02


@register_task
class WalkToTargetTask(TaskConfig):
    """Locomotion task used to pretrain approach/arrival skills for hug transfer."""

    TARGET_BODY_NAME = "walk_target"

    @property
    def name(self) -> str:
        return "walk_to_target"

    @property
    def num_curriculum_stages(self) -> int:
        return 4

    def __init__(self):
        self._weights = _WALK_STAGES[0].copy()
        self._stage = 0
        self._target_xy = np.zeros(2, dtype=np.float32)
        self._prev_dist: float | None = None
        self._fixed_standing = False
        self._control_mode = "all"

    def mjcf_worldbody_additions(self) -> str:
        return """
    <body name="walk_target" mocap="true" pos="0 0 0.02">
      <geom name="walk_target_geom" type="cylinder" size="0.16 0.005"
            rgba="0.2 0.8 0.2 0.45" contype="0" conaffinity="0"/>
      <site name="walk_target_site" type="sphere" size="0.045" rgba="0.15 0.95 0.25 1"/>
    </body>"""

    def get_contact_pairs(self) -> List[Tuple[str, str, str]]:
        return []

    def _set_target_marker_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.TARGET_BODY_NAME)
        if body_id < 0:
            return
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            return
        data.mocap_pos[mocap_id, 0] = float(self._target_xy[0])
        data.mocap_pos[mocap_id, 1] = float(self._target_xy[1])
        data.mocap_pos[mocap_id, 2] = _TARGET_Z
        data.mocap_quat[mocap_id, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def randomize_state(self, model, data, id_cache, rng):
        h0_qpos = id_cache.joint_qpos_idx["h0"]
        h1_qpos = id_cache.joint_qpos_idx["h1"]

        h0_xy = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)], dtype=np.float32)
        yaw0 = rng.uniform(-np.pi, np.pi)
        data.qpos[h0_qpos[0]] = float(h0_xy[0])
        data.qpos[h0_qpos[1]] = float(h0_xy[1])
        data.qpos[h0_qpos[2]] = DEFAULT_ROOT_Z
        data.qpos[h0_qpos[3]] = np.cos(yaw0 / 2.0)
        data.qpos[h0_qpos[4]] = 0.0
        data.qpos[h0_qpos[5]] = 0.0
        data.qpos[h0_qpos[6]] = np.sin(yaw0 / 2.0)

        # Keep the non-training partner far away so this remains a pure single-agent task.
        data.qpos[h1_qpos[0]] = 8.0 + rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[1]] = 8.0 + rng.uniform(-0.1, 0.1)
        data.qpos[h1_qpos[2]] = DEFAULT_ROOT_Z
        yaw1 = rng.uniform(-0.2, 0.2)
        data.qpos[h1_qpos[3]] = np.cos(yaw1 / 2.0)
        data.qpos[h1_qpos[4]] = 0.0
        data.qpos[h1_qpos[5]] = 0.0
        data.qpos[h1_qpos[6]] = np.sin(yaw1 / 2.0)

        d_min, d_max = _TARGET_DISTANCE_BY_STAGE.get(self._stage, _TARGET_DISTANCE_BY_STAGE[0])
        target_dist = rng.uniform(d_min, d_max)
        target_bearing = rng.uniform(-np.pi, np.pi)
        self._target_xy = np.asarray(
            [
                h0_xy[0] + target_dist * np.cos(target_bearing),
                h0_xy[1] + target_dist * np.sin(target_bearing),
            ],
            dtype=np.float32,
        )
        self._set_target_marker_pose(model, data)

        data.qvel[:] = 0.0
        self._prev_dist = None

    @property
    def task_obs_dim(self) -> int:
        # target_delta_local(3), target_dist, heading(cos/sin), root_vel_local_xy(2), speed_xy, arrival_gate
        return 10

    def _target_metrics(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        agent: str,
    ) -> Dict[str, float | np.ndarray]:
        torso_pos = id_cache.get_torso_xpos(data, agent)
        torso_xmat = id_cache.get_torso_xmat(data, agent)
        root_vel = get_root_linear_velocity(data, id_cache.joint_qvel_idx[agent])

        target_world = np.array([self._target_xy[0], self._target_xy[1], torso_pos[2]], dtype=np.float64)
        delta_world = target_world - torso_pos
        delta_local = torso_xmat.T @ delta_world
        dist_xy = float(np.linalg.norm(delta_world[:2]))

        fwd = get_forward_vector(torso_xmat)
        fwd_xy = fwd[:2]
        fwd_xy_norm = np.linalg.norm(fwd_xy)
        if fwd_xy_norm < 1e-8:
            fwd_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            fwd_xy = fwd_xy / fwd_xy_norm

        if dist_xy < 1e-8:
            dir_to_target_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            dir_to_target_xy = delta_world[:2] / dist_xy

        heading_alignment = float(np.dot(fwd_xy, dir_to_target_xy))
        heading_signed = float(fwd_xy[0] * dir_to_target_xy[1] - fwd_xy[1] * dir_to_target_xy[0])
        speed_xy = float(np.linalg.norm(root_vel[:2]))
        vel_local = torso_xmat.T @ root_vel
        tilt = float(compute_tilt_angle(get_up_vector(torso_xmat)))
        upright_score = float(np.exp(-2.5 * max(0.0, tilt)))

        return {
            "delta_local": delta_local.astype(np.float32),
            "dist_xy": dist_xy,
            "heading_alignment": heading_alignment,
            "heading_signed": heading_signed,
            "speed_xy": speed_xy,
            "vel_local": vel_local.astype(np.float32),
            "tilt": tilt,
            "upright_score": upright_score,
        }

    def compute_task_obs(self, data, id_cache, agent, contact_info):
        m = self._target_metrics(data, id_cache, agent)
        thresholds = _SUCCESS_THRESHOLDS.get(self._stage, _SUCCESS_THRESHOLDS[0])
        in_arrival_zone = 1.0 if float(m["dist_xy"]) < thresholds["arrival_dist"] else 0.0
        delta_local = m["delta_local"]
        vel_local = m["vel_local"]
        return np.asarray(
            [
                delta_local[0],
                delta_local[1],
                delta_local[2],
                m["dist_xy"],
                m["heading_alignment"],
                m["heading_signed"],
                vel_local[0],
                vel_local[1],
                m["speed_xy"],
                in_arrival_zone,
            ],
            dtype=np.float32,
        )

    def compute_reward(
        self,
        data,
        id_cache,
        contact_info,
        ctrl,
        contact_force_proxy,
        hold_steps,
        success,
        fallen,
    ) -> Tuple[float, Dict[str, Any]]:
        w = self._weights
        metrics = self._target_metrics(data, id_cache, "h0")
        thresholds = _SUCCESS_THRESHOLDS.get(self._stage, _SUCCESS_THRESHOLDS[0])

        dist_xy = float(metrics["dist_xy"])
        heading = float(metrics["heading_alignment"])
        speed_xy = float(metrics["speed_xy"])
        tilt = float(metrics["tilt"])
        upright_score = float(metrics["upright_score"])
        arrival_gate = 1.0 if dist_xy < thresholds["arrival_dist"] else 0.0
        vel_local_x = float(np.asarray(metrics["vel_local"])[0])

        if self._prev_dist is None:
            progress = 0.0
        else:
            progress = float(self._prev_dist - dist_xy)
        progress = float(np.clip(progress, -w["progress_clip"], w["progress_clip"]))
        self._prev_dist = dist_xy

        r_progress = w["w_progress"] * progress
        r_distance = w["w_distance"] * np.exp(-w["distance_scale"] * dist_xy)
        # Penalize facing away from the target (negative heading), otherwise policies can
        # optimize pure distance progress while walking "backwards" relative to the torso.
        r_heading = w["w_heading"] * float(np.clip(heading, -1.0, 1.0))
        r_arrival = w["w_arrival"] * arrival_gate
        r_stop = w["w_stop"] * arrival_gate * np.exp(-6.0 * speed_xy)
        r_face = w["w_face"] * arrival_gate * max(0.0, heading)
        r_upright = w["w_upright"] * upright_score
        r_time = float(w.get("w_time", 0.0))
        # Discourage moving opposite the torso forward axis except when already in the arrival zone.
        r_backpedal = w.get("w_backpedal", 0.0) * (1.0 - arrival_gate) * max(0.0, -vel_local_x)
        r_energy = w["w_energy"] * float(np.sum(np.square(ctrl)))
        r_impact = w["w_impact"] * float(contact_force_proxy)

        total = (
            r_progress
            + r_distance
            + r_heading
            + r_arrival
            + r_stop
            + r_face
            + r_upright
            + r_time
            + r_backpedal
            + r_energy
            + r_impact
        )

        r_fall = 0.0
        if fallen:
            r_fall = float(w["w_fall"])
            total += r_fall

        r_success = 0.0
        if success:
            r_success = float(w["r_success"])
            total += r_success

        info = {
            "walk_target_x": float(self._target_xy[0]),
            "walk_target_y": float(self._target_xy[1]),
            "walk_dist_xy": dist_xy,
            "walk_heading_alignment": heading,
            "walk_speed_xy": speed_xy,
            "walk_vel_local_x": vel_local_x,
            "walk_tilt": tilt,
            "walk_arrival_gate": arrival_gate,
            "r_progress": r_progress,
            "r_distance": r_distance,
            "r_heading": r_heading,
            "r_arrival": r_arrival,
            "r_stop": r_stop,
            "r_face": r_face,
            "r_upright": r_upright,
            "r_time": r_time,
            "r_backpedal": r_backpedal,
            "r_energy": r_energy,
            "r_impact": r_impact,
            "r_fall": r_fall,
            "r_success": r_success,
            "total_reward": float(total),
            "hold_steps": int(hold_steps),
        }
        return float(total), info

    def check_success(self, data, id_cache, contact_info):
        thresholds = _SUCCESS_THRESHOLDS.get(self._stage, _SUCCESS_THRESHOLDS[0])
        metrics = self._target_metrics(data, id_cache, "h0")

        dist_xy = float(metrics["dist_xy"])
        heading = float(metrics["heading_alignment"])
        speed_xy = float(metrics["speed_xy"])
        tilt = float(metrics["tilt"])

        dist_ok = dist_xy < thresholds["success_dist"]
        heading_ok = heading > thresholds["success_heading"]
        speed_ok = speed_xy < thresholds["success_speed"]
        upright_ok = tilt < _UPRIGHT_TILT_THRESH
        condition = dist_ok and heading_ok and speed_ok and upright_ok

        info = {
            "walk_success_dist": dist_xy,
            "walk_success_dist_thresh": thresholds["success_dist"],
            "walk_success_dist_ok": dist_ok,
            "walk_success_heading": heading,
            "walk_success_heading_thresh": thresholds["success_heading"],
            "walk_success_heading_ok": heading_ok,
            "walk_success_speed": speed_xy,
            "walk_success_speed_thresh": thresholds["success_speed"],
            "walk_success_speed_ok": speed_ok,
            "walk_success_upright_ok": upright_ok,
            "walk_success_condition": condition,
        }
        return condition, info

    def check_fallen(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
    ) -> Tuple[bool, str]:
        # This is a single-agent pretraining task; ignore h1 failures.
        torso_pos = id_cache.get_torso_xpos(data, "h0")
        torso_xmat = id_cache.get_torso_xmat(data, "h0")
        tilt = compute_tilt_angle(get_up_vector(torso_xmat))
        if torso_pos[2] < 0.5:
            return True, "h0"
        if tilt > np.pi / 2:
            return True, "h0"
        return False, ""

    def set_stage(self, stage: int) -> None:
        self._stage = int(stage)
        self._weights = _WALK_STAGES.get(self._stage, _WALK_STAGES[0]).copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self._weights.copy()
