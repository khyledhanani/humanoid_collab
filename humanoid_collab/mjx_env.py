"""MJX-backed PettingZoo environment for two-agent humanoid collaboration.

This implementation keeps physics, observations, rewards, success checks,
and contact proxies fully on-device in JAX for maximum throughput.
CPU MuJoCo data is only synchronized when rendering or querying full state.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import mujoco

from humanoid_collab.env import HumanoidCollabEnv


class _MJXState(NamedTuple):
    """JAX state carried across steps."""

    data: Any
    step_count: Any
    hold_steps: Any
    terminated: Any
    truncated: Any
    weights: Any


def _import_mjx_modules():
    """Import MJX stack lazily with clear install guidance."""
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MJX backend requires `jax` and `jaxlib`. "
            "Install with: pip install -e '.[mjx]'"
        ) from exc

    try:
        from mujoco import mjx as mujoco_mjx  # type: ignore
        mjx = mujoco_mjx
    except Exception:
        try:
            import mujoco_mjx as standalone_mjx  # type: ignore
            mjx = standalone_mjx
        except Exception as exc:
            raise ImportError(
                "MJX backend not found. Install MuJoCo MJX bindings, e.g. "
                "`pip install mujoco-mjx`."
            ) from exc

    return jax, jnp, mjx


class MJXHumanoidCollabEnv(HumanoidCollabEnv):
    """Two-agent humanoid collaboration environment with MJX backend.

    Key properties:
    - Per-step logic is fully JAX/MJX (no CPU sync in `step`).
    - CPU sync happens only in `render`, `state`, and `sync_render_state`.
    - Supports a fast path (`step_jax`) for JAX-native training loops.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "humanoid_collab_mjx_v1",
        "is_parallelizable": True,
    }

    _HUG_WEIGHT_KEYS = [
        "w_dist",
        "w_face",
        "w_stab",
        "w_contact",
        "w_hand",
        "w_energy",
        "w_impact",
        "w_fall",
        "r_success",
    ]
    _HANDSHAKE_WEIGHT_KEYS = [
        "w_dist",
        "w_face",
        "w_stab",
        "w_hand_prox",
        "w_contact",
        "w_energy",
        "w_impact",
        "w_fall",
        "r_success",
    ]
    _BOX_WEIGHT_KEYS = [
        "w_approach",
        "w_hand_box",
        "w_grip",
        "w_lift",
        "w_hold",
        "w_energy",
        "w_impact",
        "w_fall",
        "r_success",
    ]

    def __init__(
        self,
        task: str = "hug",
        render_mode: Optional[str] = None,
        horizon: int = 1000,
        frame_skip: int = 5,
        hold_target: int = 30,
        stage: int = 0,
        jit: bool = True,
        detailed_info: bool = False,
    ):
        self._jax, self._jnp, self._mjx = _import_mjx_modules()
        self._jit = bool(jit)
        self._detailed_info = bool(detailed_info)

        super().__init__(
            task=task,
            render_mode=render_mode,
            horizon=horizon,
            frame_skip=frame_skip,
            hold_target=hold_target,
            stage=stage,
        )

        self._backend = "mjx"
        self._mjx_model = self._mjx.put_model(self.model)
        self._mjx_data = self._mjx.make_data(self._mjx_model)
        self._mjx_state = None

        if self.task_name == "hug":
            self._weight_keys = self._HUG_WEIGHT_KEYS
        elif self.task_name == "handshake":
            self._weight_keys = self._HANDSHAKE_WEIGHT_KEYS
        elif self.task_name == "box_lift":
            self._weight_keys = self._BOX_WEIGHT_KEYS
        else:
            raise ValueError(f"Unsupported task for MJX backend: {self.task_name}")

        self._prepare_static_indices()
        self._build_kernels()
        self._push_cpu_state_to_mjx()
        self._init_mjx_state_from_cpu()

    def _prepare_static_indices(self) -> None:
        """Prepare static index arrays/constants consumed by JAX kernels."""
        jnp = self._jnp

        self._h0_act_idx = jnp.asarray(self.id_cache.actuator_idx["h0"], dtype=jnp.int32)
        self._h1_act_idx = jnp.asarray(self.id_cache.actuator_idx["h1"], dtype=jnp.int32)

        self._h0_qpos_idx = jnp.asarray(self.id_cache.joint_qpos_idx["h0"], dtype=jnp.int32)
        self._h1_qpos_idx = jnp.asarray(self.id_cache.joint_qpos_idx["h1"], dtype=jnp.int32)
        self._h0_qvel_idx = jnp.asarray(self.id_cache.joint_qvel_idx["h0"], dtype=jnp.int32)
        self._h1_qvel_idx = jnp.asarray(self.id_cache.joint_qvel_idx["h1"], dtype=jnp.int32)

        self._h0_torso_body_id = int(self.id_cache.torso_body_ids["h0"])
        self._h1_torso_body_id = int(self.id_cache.torso_body_ids["h1"])

        def sid(name: str) -> int:
            v = self.id_cache.site_ids.get(name)
            if v is None:
                return -1
            return int(v)

        self._sid = {
            "h0_chest": sid("h0_chest"),
            "h1_chest": sid("h1_chest"),
            "h0_pelvis": sid("h0_pelvis"),
            "h1_pelvis": sid("h1_pelvis"),
            "h0_back": sid("h0_back"),
            "h1_back": sid("h1_back"),
            "h0_lhand": sid("h0_lhand"),
            "h0_rhand": sid("h0_rhand"),
            "h1_lhand": sid("h1_lhand"),
            "h1_rhand": sid("h1_rhand"),
            "box_center": sid("box_center"),
        }

        self._box_qvel_start = -1
        self._box_qpos_start = -1
        for i in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name == "box_joint":
                self._box_qvel_start = int(self.model.jnt_dofadr[i])
                self._box_qpos_start = int(self.model.jnt_qposadr[i])
                break

        # Contact proxy groups resolved once, then used in JAX kernels.
        self._contact_pair_keys: List[str] = []
        self._contact_pair_geom_a = []
        self._contact_pair_geom_b = []
        self._contact_pair_thresholds: List[float] = []

        for key, group_a, group_b in self.task_config.get_contact_pairs():
            geoms_a = sorted(self.id_cache.get_geom_group(group_a))
            geoms_b = sorted(self.id_cache.get_geom_group(group_b))
            if len(geoms_a) == 0 or len(geoms_b) == 0:
                raise ValueError(f"Empty geom group in contact pair: {key} ({group_a}, {group_b})")
            self._contact_pair_keys.append(key)
            self._contact_pair_geom_a.append(jnp.asarray(np.array(geoms_a, dtype=np.int32)))
            self._contact_pair_geom_b.append(jnp.asarray(np.array(geoms_b, dtype=np.int32)))
            self._contact_pair_thresholds.append(self._contact_threshold_for_key(key))

        self._contact_idx = {k: i for i, k in enumerate(self._contact_pair_keys)}

    @staticmethod
    def _contact_threshold_for_key(key: str) -> float:
        """Heuristic distance threshold for proxy contact checks."""
        if "hand" in key and "box" in key:
            return 0.20
        if "hand" in key and key.count("hand") >= 2:
            return 0.14
        if "arm" in key and "torso" in key:
            return 0.20
        return 0.18

    def _encode_weights(self) -> np.ndarray:
        weights = self.task_config.get_weights_dict()
        return np.asarray([weights[k] for k in self._weight_keys], dtype=np.float32)

    def _push_cpu_state_to_mjx(self) -> None:
        qpos = self._jnp.asarray(self.data.qpos)
        qvel = self._jnp.asarray(self.data.qvel)
        ctrl = self._jnp.asarray(self.data.ctrl)
        self._mjx_data = self._mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        self._mjx_data = self._mjx_forward_kernel(self._mjx_data)

    def _pull_mjx_state_to_cpu(self) -> None:
        if self._mjx_state is None:
            return
        qpos, qvel, ctrl = self._jax.device_get(
            (self._mjx_state.data.qpos, self._mjx_state.data.qvel, self._mjx_state.data.ctrl)
        )
        self.data.qpos[:] = np.asarray(qpos)
        self.data.qvel[:] = np.asarray(qvel)
        self.data.ctrl[:] = np.asarray(ctrl)
        mujoco.mj_forward(self.model, self.data)

    def _init_mjx_state_from_cpu(self) -> None:
        jnp = self._jnp
        weights = jnp.asarray(self._encode_weights(), dtype=jnp.float32)
        self._mjx_state = _MJXState(
            data=self._mjx_data,
            step_count=jnp.asarray(self._step_count, dtype=jnp.int32),
            hold_steps=jnp.asarray(self._hold_steps, dtype=jnp.int32),
            terminated=jnp.asarray(self._terminated, dtype=jnp.bool_),
            truncated=jnp.asarray(self._truncated, dtype=jnp.bool_),
            weights=weights,
        )

    def _build_kernels(self) -> None:
        """Build JAX/MJX kernels for observe and step."""
        jax = self._jax
        jnp = self._jnp
        mjx = self._mjx
        mjx_model = self._mjx_model

        h0_act_idx = self._h0_act_idx
        h1_act_idx = self._h1_act_idx
        h0_qpos_idx = self._h0_qpos_idx
        h1_qpos_idx = self._h1_qpos_idx
        h0_qvel_idx = self._h0_qvel_idx
        h1_qvel_idx = self._h1_qvel_idx
        h0_torso_body_id = self._h0_torso_body_id
        h1_torso_body_id = self._h1_torso_body_id

        sid = self._sid
        box_qvel_start = self._box_qvel_start

        contact_geom_a = self._contact_pair_geom_a
        contact_geom_b = self._contact_pair_geom_b
        contact_thresholds = self._contact_pair_thresholds
        contact_idx = self._contact_idx

        frame_skip = int(self.frame_skip)
        hold_target = int(self.hold_target)
        horizon = int(self.horizon)
        action_dim = int(self._action_dim)

        def normalize(v):
            return v / (jnp.linalg.norm(v) + 1e-8)

        def torso_pose(data, body_id):
            pos = data.xpos[body_id]
            xmat = data.xmat[body_id].reshape(3, 3)
            return pos, xmat

        def forward_vec(xmat):
            return normalize(xmat[:, 0])

        def up_vec(xmat):
            return normalize(xmat[:, 2])

        def tilt_from_up(u):
            world_up = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)
            c = jnp.clip(jnp.dot(u, world_up), -1.0, 1.0)
            return jnp.arccos(c)

        def root_linvel(data, qvel_idx):
            return data.qvel[qvel_idx[0:3]]

        def root_angvel(data, qvel_idx):
            return data.qvel[qvel_idx[3:6]]

        def rotate_local(vec_world, xmat):
            return xmat.T @ vec_world

        def pairwise_min_dist(data, ids_a, ids_b):
            pa = data.geom_xpos[ids_a]
            pb = data.geom_xpos[ids_b]
            d = jnp.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
            return jnp.min(d), d

        def compute_contacts(data):
            flags = []
            force_proxy = jnp.asarray(0.0, dtype=jnp.float32)
            for ids_a, ids_b, thresh in zip(contact_geom_a, contact_geom_b, contact_thresholds):
                min_d, dmat = pairwise_min_dist(data, ids_a, ids_b)
                penetration = jnp.maximum(jnp.asarray(thresh, dtype=jnp.float32) - dmat, 0.0)
                flags.append(min_d < thresh)
                force_proxy = force_proxy + 1000.0 * jnp.sum(penetration)
            return jnp.asarray(flags, dtype=jnp.bool_), force_proxy

        def cflag(flags, key):
            return flags[contact_idx[key]]

        def build_base_obs(data, agent):
            if agent == "h0":
                qpos_idx = h0_qpos_idx
                qvel_idx = h0_qvel_idx
                partner_qvel_idx = h1_qvel_idx
                self_torso_id = h0_torso_body_id
                partner_torso_id = h1_torso_body_id
                partner_chest_sid = sid["h1_chest"]
                partner_pelvis_sid = sid["h1_pelvis"]
            else:
                qpos_idx = h1_qpos_idx
                qvel_idx = h1_qvel_idx
                partner_qvel_idx = h0_qvel_idx
                self_torso_id = h1_torso_body_id
                partner_torso_id = h0_torso_body_id
                partner_chest_sid = sid["h0_chest"]
                partner_pelvis_sid = sid["h0_pelvis"]

            self_pos, self_xmat = torso_pose(data, self_torso_id)
            partner_pos, partner_xmat = torso_pose(data, partner_torso_id)

            joint_qpos = data.qpos[qpos_idx[7:]]
            joint_qvel = data.qvel[qvel_idx[6:]]
            root_quat = data.qpos[qpos_idx[3:7]]
            self_angvel = root_angvel(data, qvel_idx)

            partner_chest = data.site_xpos[partner_chest_sid]
            rel_chest_local = rotate_local(partner_chest - self_pos, self_xmat)

            partner_pelvis = data.site_xpos[partner_pelvis_sid]
            rel_pelvis_local = rotate_local(partner_pelvis - self_pos, self_xmat)

            partner_linvel = root_linvel(data, partner_qvel_idx)
            self_lin = root_linvel(data, qvel_idx)
            rel_vel_local = rotate_local(partner_linvel - self_lin, self_xmat)

            facing = jnp.dot(forward_vec(self_xmat), -forward_vec(partner_xmat))
            up_align = jnp.dot(up_vec(self_xmat), up_vec(partner_xmat))

            return jnp.concatenate(
                [
                    joint_qpos,
                    joint_qvel,
                    root_quat,
                    self_angvel,
                    rel_chest_local,
                    rel_pelvis_local,
                    rel_vel_local,
                    jnp.asarray([facing, up_align], dtype=jnp.float32),
                ],
                axis=0,
            )

        task_name = self.task_name

        # These indices are task-specific and only used by the relevant branch.
        idx_h0_arm_h1_torso = contact_idx.get("h0_arm_h1_torso", -1)
        idx_h1_arm_h0_torso = contact_idx.get("h1_arm_h0_torso", -1)
        idx_h0_l_arm_h1_torso = contact_idx.get("h0_l_arm_h1_torso", -1)
        idx_h0_r_arm_h1_torso = contact_idx.get("h0_r_arm_h1_torso", -1)
        idx_h1_l_arm_h0_torso = contact_idx.get("h1_l_arm_h0_torso", -1)
        idx_h1_r_arm_h0_torso = contact_idx.get("h1_r_arm_h0_torso", -1)
        idx_h0_hand_h1_hand = contact_idx.get("h0_hand_h1_hand", -1)
        idx_h0_r_hand_h1_r_hand = contact_idx.get("h0_r_hand_h1_r_hand", -1)
        idx_h0_l_hand_box = contact_idx.get("h0_l_hand_box", -1)
        idx_h0_r_hand_box = contact_idx.get("h0_r_hand_box", -1)
        idx_h1_l_hand_box = contact_idx.get("h1_l_hand_box", -1)
        idx_h1_r_hand_box = contact_idx.get("h1_r_hand_box", -1)
        idx_h0_hand_box = contact_idx.get("h0_hand_box", -1)
        idx_h1_hand_box = contact_idx.get("h1_hand_box", -1)

        def build_task_obs(data, contact_flags, agent):
            if task_name == "hug":
                if agent == "h0":
                    l_arm = contact_flags[idx_h0_l_arm_h1_torso]
                    r_arm = contact_flags[idx_h0_r_arm_h1_torso]
                    partner_arm = contact_flags[idx_h1_arm_h0_torso]
                else:
                    l_arm = contact_flags[idx_h1_l_arm_h0_torso]
                    r_arm = contact_flags[idx_h1_r_arm_h0_torso]
                    partner_arm = contact_flags[idx_h0_arm_h1_torso]
                return jnp.asarray([l_arm, r_arm, partner_arm], dtype=jnp.float32)

            if task_name == "handshake":
                if agent == "h0":
                    self_torso_id = h0_torso_body_id
                    partner_rhand_sid = sid["h1_rhand"]
                else:
                    self_torso_id = h1_torso_body_id
                    partner_rhand_sid = sid["h0_rhand"]
                self_pos, self_xmat = torso_pose(data, self_torso_id)
                partner_rhand = data.site_xpos[partner_rhand_sid]
                rel_rhand_local = rotate_local(partner_rhand - self_pos, self_xmat)
                any_contact = contact_flags[idx_h0_hand_h1_hand]
                my_r_partner_r = contact_flags[idx_h0_r_hand_h1_r_hand]
                return jnp.concatenate(
                    [
                        rel_rhand_local,
                        jnp.asarray([any_contact, my_r_partner_r], dtype=jnp.float32),
                    ],
                    axis=0,
                )

            # box_lift
            if agent == "h0":
                self_torso_id = h0_torso_body_id
                l_hand_box = contact_flags[idx_h0_l_hand_box]
                r_hand_box = contact_flags[idx_h0_r_hand_box]
                partner_hand_box = contact_flags[idx_h1_hand_box]
            else:
                self_torso_id = h1_torso_body_id
                l_hand_box = contact_flags[idx_h1_l_hand_box]
                r_hand_box = contact_flags[idx_h1_r_hand_box]
                partner_hand_box = contact_flags[idx_h0_hand_box]

            self_pos, self_xmat = torso_pose(data, self_torso_id)
            box_pos = data.site_xpos[sid["box_center"]]
            rel_box_local = rotate_local(box_pos - self_pos, self_xmat)
            box_height = box_pos[2]

            if box_qvel_start >= 0:
                box_vel = data.qvel[box_qvel_start : box_qvel_start + 3]
                box_vel_mag = jnp.linalg.norm(box_vel)
            else:
                box_vel_mag = jnp.asarray(0.0, dtype=jnp.float32)

            return jnp.concatenate(
                [
                    rel_box_local,
                    jnp.asarray(
                        [box_height, l_hand_box, r_hand_box, partner_hand_box, box_vel_mag],
                        dtype=jnp.float32,
                    ),
                ],
                axis=0,
            )

        def build_full_obs(data, contact_flags):
            h0_base = build_base_obs(data, "h0")
            h1_base = build_base_obs(data, "h1")
            h0_task = build_task_obs(data, contact_flags, "h0")
            h1_task = build_task_obs(data, contact_flags, "h1")
            h0_obs = jnp.concatenate([h0_base, h0_task], axis=0).astype(jnp.float32)
            h1_obs = jnp.concatenate([h1_base, h1_task], axis=0).astype(jnp.float32)
            return jnp.stack([h0_obs, h1_obs], axis=0)

        def check_fallen(data):
            h0_pos, h0_xmat = torso_pose(data, h0_torso_body_id)
            h1_pos, h1_xmat = torso_pose(data, h1_torso_body_id)
            h0_tilt = tilt_from_up(up_vec(h0_xmat))
            h1_tilt = tilt_from_up(up_vec(h1_xmat))
            h0_fallen = (h0_pos[2] < 0.5) | (h0_tilt > (jnp.pi / 2.0))
            h1_fallen = (h1_pos[2] < 0.5) | (h1_tilt > (jnp.pi / 2.0))
            fallen = h0_fallen | h1_fallen
            fallen_code = jnp.where(h0_fallen, jnp.int32(0), jnp.where(h1_fallen, jnp.int32(1), jnp.int32(-1)))
            return fallen, fallen_code

        def task_success_and_reward(data, contact_flags, ctrl, force_proxy, hold_steps_prev, weights):
            if task_name == "hug":
                h0_chest = data.site_xpos[sid["h0_chest"]]
                h1_chest = data.site_xpos[sid["h1_chest"]]
                dist = jnp.linalg.norm(h0_chest - h1_chest)

                h0_pos, h0_xmat = torso_pose(data, h0_torso_body_id)
                h1_pos, h1_xmat = torso_pose(data, h1_torso_body_id)
                facing = jnp.dot(forward_vec(h0_xmat), -forward_vec(h1_xmat))

                h0_vel = root_linvel(data, h0_qvel_idx)
                h1_vel = root_linvel(data, h1_qvel_idx)
                rel_speed = jnp.linalg.norm(h0_vel - h1_vel)

                contact_ok = contact_flags[idx_h0_arm_h1_torso] & contact_flags[idx_h1_arm_h0_torso]
                dist_ok = (dist > 0.25) & (dist < 0.60)
                facing_ok = facing > 0.6
                speed_ok = rel_speed < 1.0
                upright_ok = (tilt_from_up(up_vec(h0_xmat)) < 0.5) & (tilt_from_up(up_vec(h1_xmat)) < 0.5)
                success_condition = dist_ok & facing_ok & contact_ok & speed_ok & upright_ok

                n_contacts = contact_flags[idx_h0_arm_h1_torso].astype(jnp.float32) + contact_flags[idx_h1_arm_h0_torso].astype(jnp.float32)

                r_dist = weights[0] * jnp.exp(-3.0 * dist)
                r_face = weights[1] * jnp.maximum(0.0, facing)
                r_stab = weights[2] * jnp.exp(-2.0 * rel_speed)
                r_contact = weights[3] * n_contacts

                h0l = data.site_xpos[sid["h0_lhand"]]
                h0r = data.site_xpos[sid["h0_rhand"]]
                h1l = data.site_xpos[sid["h1_lhand"]]
                h1r = data.site_xpos[sid["h1_rhand"]]
                h0b = data.site_xpos[sid["h0_back"]]
                h1b = data.site_xpos[sid["h1_back"]]
                hand_r = (
                    jnp.exp(-4.0 * jnp.linalg.norm(h0l - h1b))
                    + jnp.exp(-4.0 * jnp.linalg.norm(h0r - h1b))
                    + jnp.exp(-4.0 * jnp.linalg.norm(h1l - h0b))
                    + jnp.exp(-4.0 * jnp.linalg.norm(h1r - h0b))
                )
                r_hand = weights[4] * hand_r

                reward_no_term = (
                    r_dist
                    + r_face
                    + r_stab
                    + r_contact
                    + r_hand
                    + weights[5] * jnp.sum(jnp.square(ctrl))
                    + weights[6] * force_proxy
                )
                return success_condition, reward_no_term

            if task_name == "handshake":
                h0_chest = data.site_xpos[sid["h0_chest"]]
                h1_chest = data.site_xpos[sid["h1_chest"]]
                dist = jnp.linalg.norm(h0_chest - h1_chest)
                dist_error = jnp.abs(dist - 0.7)

                h0_pos, h0_xmat = torso_pose(data, h0_torso_body_id)
                h1_pos, h1_xmat = torso_pose(data, h1_torso_body_id)
                facing = jnp.dot(forward_vec(h0_xmat), -forward_vec(h1_xmat))

                h0_vel = root_linvel(data, h0_qvel_idx)
                h1_vel = root_linvel(data, h1_qvel_idx)
                rel_speed = jnp.linalg.norm(h0_vel - h1_vel)

                contact_ok = contact_flags[idx_h0_hand_h1_hand]
                dist_ok = (dist > 0.4) & (dist < 1.0)
                facing_ok = facing > 0.6
                speed_ok = rel_speed < 0.8
                upright_ok = (tilt_from_up(up_vec(h0_xmat)) < 0.5) & (tilt_from_up(up_vec(h1_xmat)) < 0.5)
                success_condition = dist_ok & facing_ok & contact_ok & speed_ok & upright_ok

                h0_rhand = data.site_xpos[sid["h0_rhand"]]
                h1_rhand = data.site_xpos[sid["h1_rhand"]]
                hand_dist = jnp.linalg.norm(h0_rhand - h1_rhand)

                r_dist = weights[0] * jnp.exp(-3.0 * dist_error)
                r_face = weights[1] * jnp.maximum(0.0, facing)
                r_stab = weights[2] * jnp.exp(-2.0 * rel_speed)
                r_hand = weights[3] * jnp.exp(-4.0 * hand_dist)
                r_contact = weights[4] * contact_ok.astype(jnp.float32)
                reward_no_term = (
                    r_dist
                    + r_face
                    + r_stab
                    + r_hand
                    + r_contact
                    + weights[5] * jnp.sum(jnp.square(ctrl))
                    + weights[6] * force_proxy
                )
                return success_condition, reward_no_term

            # box_lift
            box_pos = data.site_xpos[sid["box_center"]]
            box_height = box_pos[2]
            if box_qvel_start >= 0:
                box_vel = data.qvel[box_qvel_start : box_qvel_start + 3]
                box_vel_mag = jnp.linalg.norm(box_vel)
            else:
                box_vel_mag = jnp.asarray(0.0, dtype=jnp.float32)

            h0_pos, h0_xmat = torso_pose(data, h0_torso_body_id)
            h1_pos, h1_xmat = torso_pose(data, h1_torso_body_id)

            height_ok = box_height > 0.8
            both_gripping = contact_flags[idx_h0_hand_box] & contact_flags[idx_h1_hand_box]
            stable = box_vel_mag < 0.5
            upright_ok = (tilt_from_up(up_vec(h0_xmat)) < 0.5) & (tilt_from_up(up_vec(h1_xmat)) < 0.5)
            success_condition = height_ok & both_gripping & stable & upright_ok

            dist_h0 = jnp.linalg.norm(h0_pos[:2] - box_pos[:2])
            dist_h1 = jnp.linalg.norm(h1_pos[:2] - box_pos[:2])
            r_approach = weights[0] * (jnp.exp(-3.0 * dist_h0) + jnp.exp(-3.0 * dist_h1))

            h0l = data.site_xpos[sid["h0_lhand"]]
            h0r = data.site_xpos[sid["h0_rhand"]]
            h1l = data.site_xpos[sid["h1_lhand"]]
            h1r = data.site_xpos[sid["h1_rhand"]]
            r_hand_box = weights[1] * (
                jnp.exp(-4.0 * jnp.linalg.norm(h0l - box_pos))
                + jnp.exp(-4.0 * jnp.linalg.norm(h0r - box_pos))
                + jnp.exp(-4.0 * jnp.linalg.norm(h1l - box_pos))
                + jnp.exp(-4.0 * jnp.linalg.norm(h1r - box_pos))
            )

            n_grips = contact_flags[idx_h0_hand_box].astype(jnp.float32) + contact_flags[idx_h1_hand_box].astype(jnp.float32)
            r_grip = weights[2] * n_grips

            lift_progress = jnp.minimum(1.0, box_height / 0.8)
            r_lift = weights[3] * lift_progress
            r_hold = weights[4] * jnp.where(box_height > 0.72, jnp.exp(-2.0 * box_vel_mag), 0.0)

            reward_no_term = (
                r_approach
                + r_hand_box
                + r_grip
                + r_lift
                + r_hold
                + weights[5] * jnp.sum(jnp.square(ctrl))
                + weights[6] * force_proxy
            )
            return success_condition, reward_no_term

        def observe_kernel(data):
            contact_flags, _ = compute_contacts(data)
            obs_pair = build_full_obs(data, contact_flags)
            return obs_pair, contact_flags

        def step_kernel(state, action_h0, action_h1):
            action_h0 = jnp.clip(action_h0, -1.0, 1.0)
            action_h1 = jnp.clip(action_h1, -1.0, 1.0)

            ctrl = jnp.zeros((mjx_model.nu,), dtype=jnp.float32)
            ctrl = ctrl.at[h0_act_idx].set(action_h0[:action_dim])
            ctrl = ctrl.at[h1_act_idx].set(action_h1[:action_dim])

            data = state.data.replace(ctrl=ctrl)

            def body(_, d):
                return mjx.step(mjx_model, d)

            data = jax.lax.fori_loop(0, frame_skip, body, data)

            contact_flags, force_proxy = compute_contacts(data)
            obs_pair = build_full_obs(data, contact_flags)

            success_condition, reward_no_term = task_success_and_reward(
                data, contact_flags, ctrl, force_proxy, state.hold_steps, state.weights
            )
            hold_steps = jnp.where(success_condition, state.hold_steps + 1, jnp.int32(0))
            success = hold_steps >= hold_target

            fallen, fallen_code = check_fallen(data)
            step_count = state.step_count + 1
            terminated = success | fallen
            truncated = step_count >= horizon

            reward = reward_no_term
            reward = reward + jnp.where(fallen, state.weights[7], 0.0)
            reward = reward + jnp.where(success, state.weights[8], 0.0)

            reason_code = jnp.int32(0)
            reason_code = jnp.where(success, jnp.int32(1), reason_code)
            reason_code = jnp.where(fallen & (fallen_code == 0), jnp.int32(2), reason_code)
            reason_code = jnp.where(fallen & (fallen_code == 1), jnp.int32(3), reason_code)
            reason_code = jnp.where((~terminated) & truncated, jnp.int32(4), reason_code)

            nan_flag = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
            reward = jnp.where(nan_flag, jnp.asarray(-100.0, dtype=jnp.float32), reward)
            terminated = terminated | nan_flag
            reason_code = jnp.where(nan_flag, jnp.int32(5), reason_code)
            obs_pair = jnp.where(nan_flag, jnp.zeros_like(obs_pair), obs_pair)

            next_state = _MJXState(
                data=data,
                step_count=step_count,
                hold_steps=hold_steps,
                terminated=terminated,
                truncated=truncated,
                weights=state.weights,
            )

            return next_state, obs_pair, reward, terminated, truncated, hold_steps, reason_code, contact_flags

        def step_fast_kernel(state, action_h0, action_h1):
            """Fast transition kernel without observation/materialization outputs."""
            action_h0 = jnp.clip(action_h0, -1.0, 1.0)
            action_h1 = jnp.clip(action_h1, -1.0, 1.0)

            ctrl = jnp.zeros((mjx_model.nu,), dtype=jnp.float32)
            ctrl = ctrl.at[h0_act_idx].set(action_h0[:action_dim])
            ctrl = ctrl.at[h1_act_idx].set(action_h1[:action_dim])

            data = state.data.replace(ctrl=ctrl)

            def body(_, d):
                return mjx.step(mjx_model, d)

            data = jax.lax.fori_loop(0, frame_skip, body, data)

            contact_flags, force_proxy = compute_contacts(data)
            success_condition, reward_no_term = task_success_and_reward(
                data, contact_flags, ctrl, force_proxy, state.hold_steps, state.weights
            )
            hold_steps = jnp.where(success_condition, state.hold_steps + 1, jnp.int32(0))
            success = hold_steps >= hold_target

            fallen, _ = check_fallen(data)
            step_count = state.step_count + 1
            terminated = success | fallen
            truncated = step_count >= horizon

            reward = reward_no_term
            reward = reward + jnp.where(fallen, state.weights[7], 0.0)
            reward = reward + jnp.where(success, state.weights[8], 0.0)

            nan_flag = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
            reward = jnp.where(nan_flag, jnp.asarray(-100.0, dtype=jnp.float32), reward)
            terminated = terminated | nan_flag

            next_state = _MJXState(
                data=data,
                step_count=step_count,
                hold_steps=hold_steps,
                terminated=terminated,
                truncated=truncated,
                weights=state.weights,
            )

            return next_state, reward, terminated, truncated

        def rollout_kernel(state, actions_h0, actions_h1):
            """Scan many transitions with no host sync and no per-step payloads."""

            def body_fn(carry, xs):
                a0_t, a1_t = xs
                next_state, _, _, _ = step_fast_kernel(carry, a0_t, a1_t)
                return next_state, ()

            final_state, _ = jax.lax.scan(body_fn, state, (actions_h0, actions_h1))
            return final_state

        v_step_fast_kernel = jax.vmap(
            step_fast_kernel, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0)
        )

        def rollout_batched_kernel(state_batched, actions_h0_batched, actions_h1_batched):
            """Scan many transitions over many envs: actions shape [B, T, A]."""
            actions_h0_tba = jnp.swapaxes(actions_h0_batched, 0, 1)
            actions_h1_tba = jnp.swapaxes(actions_h1_batched, 0, 1)

            def body_fn(carry, xs):
                a0_t, a1_t = xs  # [B, A]
                next_state, _, _, _ = v_step_fast_kernel(carry, a0_t, a1_t)
                return next_state, ()

            final_state, _ = jax.lax.scan(body_fn, state_batched, (actions_h0_tba, actions_h1_tba))
            return final_state

        def forward_kernel(data):
            return mjx.forward(mjx_model, data)

        if self._jit:
            self._mjx_forward_kernel = jax.jit(forward_kernel)
            self._observe_kernel = jax.jit(observe_kernel)
            self._step_kernel = jax.jit(step_kernel)
            self._rollout_kernel = jax.jit(rollout_kernel)
            self._rollout_batched_kernel = jax.jit(rollout_batched_kernel)
        else:
            self._mjx_forward_kernel = forward_kernel
            self._observe_kernel = observe_kernel
            self._step_kernel = step_kernel
            self._rollout_kernel = rollout_kernel
            self._rollout_batched_kernel = rollout_batched_kernel

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        _, infos = super().reset(seed=seed, options=options)

        self._push_cpu_state_to_mjx()
        self._init_mjx_state_from_cpu()

        obs_pair, contact_flags = self._observe_kernel(self._mjx_state.data)
        obs_pair_np = self._jax.device_get(obs_pair)
        observations = {
            "h0": np.asarray(obs_pair_np[0], dtype=np.float32),
            "h1": np.asarray(obs_pair_np[1], dtype=np.float32),
        }

        if self._detailed_info:
            contact_flags_np = self._jax.device_get(contact_flags)
            for agent in self.possible_agents:
                for i, key in enumerate(self._contact_pair_keys):
                    infos[agent][key] = bool(contact_flags_np[i])

        for agent in self.possible_agents:
            infos[agent]["backend"] = self._backend
            infos[agent]["jit"] = self._jit
            infos[agent]["on_device_step"] = True

        return observations, infos

    def step_jax(self, action_h0, action_h1):
        """Fast JAX-native step that returns JAX arrays."""
        if self._mjx_state is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        self._mjx_state, obs_pair, reward, terminated, truncated, hold_steps, reason_code, contact_flags = self._step_kernel(
            self._mjx_state, action_h0, action_h1
        )
        return obs_pair, reward, terminated, truncated, hold_steps, reason_code, contact_flags

    def get_jax_state(self):
        """Return current internal JAX state."""
        if self._mjx_state is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        return self._mjx_state

    def set_jax_state(self, state, sync_python: bool = False) -> None:
        """Replace internal JAX state."""
        self._mjx_state = state
        if sync_python:
            step_count_np, hold_steps_np, terminated_np, truncated_np = self._jax.device_get(
                (
                    state.step_count,
                    state.hold_steps,
                    state.terminated,
                    state.truncated,
                )
            )
            self._step_count = int(step_count_np)
            self._hold_steps = int(hold_steps_np)
            self._terminated = bool(terminated_np)
            self._truncated = bool(truncated_np)
            self.agents = [] if (self._terminated or self._truncated) else self.possible_agents.copy()

    def make_batched_state(self, batch_size: int, state=None):
        """Broadcast a single state to a batched state PyTree."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        base_state = self.get_jax_state() if state is None else state

        def _tile(x):
            if hasattr(x, "shape"):
                return self._jnp.broadcast_to(x, (batch_size,) + x.shape)
            return x

        return self._jax.tree_util.tree_map(_tile, base_state)

    def rollout_jax(self, actions_h0, actions_h1, state=None, commit: bool = False):
        """Run a compiled on-device rollout.

        Args:
            actions_h0: JAX/NumPy array [T, action_dim] for agent h0.
            actions_h1: JAX/NumPy array [T, action_dim] for agent h1.
            state: Optional starting JAX state. Defaults to current env state.
            commit: If True, write final state back into this env.
        Returns:
            Final JAX state after all rollout steps.
        """
        if state is None:
            state = self.get_jax_state()

        a0 = self._jnp.asarray(actions_h0, dtype=self._jnp.float32)
        a1 = self._jnp.asarray(actions_h1, dtype=self._jnp.float32)

        if a0.ndim != 2 or a1.ndim != 2:
            raise ValueError("actions_h0/actions_h1 must have shape [T, action_dim]")
        if a0.shape != a1.shape:
            raise ValueError(f"Action shapes must match, got {a0.shape} vs {a1.shape}")
        if a0.shape[1] != self._action_dim:
            raise ValueError(f"Expected action_dim={self._action_dim}, got {a0.shape[1]}")

        final_state = self._rollout_kernel(state, a0, a1)
        if commit:
            self.set_jax_state(final_state, sync_python=True)
        return final_state

    def rollout_jax_batched(self, actions_h0, actions_h1, state_batched=None):
        """Run compiled on-device rollout over batched environments.

        Args:
            actions_h0: Array [B, T, action_dim]
            actions_h1: Array [B, T, action_dim]
            state_batched: Optional batched state. If None, tiles current state across B.
        Returns:
            Final batched JAX state.
        """
        a0 = self._jnp.asarray(actions_h0, dtype=self._jnp.float32)
        a1 = self._jnp.asarray(actions_h1, dtype=self._jnp.float32)

        if a0.ndim != 3 or a1.ndim != 3:
            raise ValueError("actions_h0/actions_h1 must have shape [B, T, action_dim]")
        if a0.shape != a1.shape:
            raise ValueError(f"Action shapes must match, got {a0.shape} vs {a1.shape}")
        if a0.shape[2] != self._action_dim:
            raise ValueError(f"Expected action_dim={self._action_dim}, got {a0.shape[2]}")

        batch_size = int(a0.shape[0])
        if state_batched is None:
            state_batched = self.make_batched_state(batch_size=batch_size)

        return self._rollout_batched_kernel(state_batched, a0, a1)

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if self._terminated or self._truncated:
            return self._get_terminal_returns()

        a0 = np.zeros((self._action_dim,), dtype=np.float32)
        a1 = np.zeros((self._action_dim,), dtype=np.float32)
        if "h0" in actions:
            a0 = np.asarray(actions["h0"], dtype=np.float32)
        if "h1" in actions:
            a1 = np.asarray(actions["h1"], dtype=np.float32)

        step_out = self._step_kernel(self._mjx_state, self._jnp.asarray(a0), self._jnp.asarray(a1))
        self._mjx_state = step_out[0]

        if self._detailed_info:
            obs_pair_np, reward_np, terminated_np, truncated_np, hold_steps_np, reason_code_np, contact_flags_np, step_count_np = self._jax.device_get(
                (
                    step_out[1],
                    step_out[2],
                    step_out[3],
                    step_out[4],
                    step_out[5],
                    step_out[6],
                    step_out[7],
                    self._mjx_state.step_count,
                )
            )
        else:
            obs_pair_np, reward_np, terminated_np, truncated_np, hold_steps_np, reason_code_np, step_count_np = self._jax.device_get(
                (
                    step_out[1],
                    step_out[2],
                    step_out[3],
                    step_out[4],
                    step_out[5],
                    step_out[6],
                    self._mjx_state.step_count,
                )
            )
            contact_flags_np = None

        self._step_count = int(step_count_np)
        self._hold_steps = int(hold_steps_np)
        self._terminated = bool(terminated_np)
        self._truncated = bool(truncated_np)

        observations = {
            "h0": np.asarray(obs_pair_np[0], dtype=np.float32),
            "h1": np.asarray(obs_pair_np[1], dtype=np.float32),
        }
        reward_value = float(reward_np)
        rewards = {"h0": reward_value, "h1": reward_value}
        terminations = {"h0": self._terminated, "h1": self._terminated}
        truncations = {"h0": self._truncated, "h1": self._truncated}

        reason_map = {
            0: None,
            1: "success",
            2: "fall_h0",
            3: "fall_h1",
            4: "horizon",
            5: "nan_error",
        }
        termination_reason = reason_map.get(int(reason_code_np))

        infos: Dict[str, Dict[str, Any]] = {}
        for agent in self.agents:
            info = {
                "step": self._step_count,
                "hold_steps": self._hold_steps,
                "termination_reason": termination_reason,
                "task": self.task_name,
                "stage": self.stage,
                "backend": self._backend,
                "jit": self._jit,
                "total_reward": reward_value,
            }
            if self._detailed_info and contact_flags_np is not None:
                for i, key in enumerate(self._contact_pair_keys):
                    info[key] = bool(contact_flags_np[i])
            infos[agent] = info

        if self._terminated or self._truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def sync_render_state(self) -> None:
        """Force CPU render state sync from MJX state."""
        self._pull_mjx_state_to_cpu()

    def render(self) -> Optional[np.ndarray]:
        self._pull_mjx_state_to_cpu()
        return super().render()

    def state(self) -> np.ndarray:
        if self._mjx_state is None:
            return super().state()
        qpos, qvel = self._jax.device_get((self._mjx_state.data.qpos, self._mjx_state.data.qvel))
        return np.concatenate([np.asarray(qpos), np.asarray(qvel)])
