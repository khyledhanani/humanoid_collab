"""Tests for fixed-standing and arms-only control modes."""

import numpy as np
import mujoco

from humanoid_collab.env import HumanoidCollabEnv


def test_arms_only_action_dim():
    env = HumanoidCollabEnv(task="handshake", control_mode="arms_only")
    try:
        assert env.action_space("h0").shape[0] == 6
        assert env.action_space("h1").shape[0] == 6
    finally:
        env.close()


def test_arms_only_controls_only_arm_actuators():
    env = HumanoidCollabEnv(task="handshake", control_mode="arms_only")
    try:
        env.reset(seed=123)
        a = np.ones((env.action_space("h0").shape[0],), dtype=np.float32)
        env.step({"h0": a, "h1": a})

        h0_all = set(env.id_cache.actuator_idx["h0"].tolist())
        h1_all = set(env.id_cache.actuator_idx["h1"].tolist())
        h0_arm = set(env._control_actuator_idx["h0"].tolist())
        h1_arm = set(env._control_actuator_idx["h1"].tolist())
        uncontrolled = sorted((h0_all - h0_arm) | (h1_all - h1_arm))

        assert np.allclose(env.data.ctrl[uncontrolled], 0.0)
    finally:
        env.close()


def test_fixed_standing_adds_weld_constraints():
    env = HumanoidCollabEnv(task="handshake", fixed_standing=True)
    try:
        assert env.model.neq >= 2
    finally:
        env.close()


def test_handshake_fixed_standing_spawn_is_reachable():
    env = HumanoidCollabEnv(
        task="handshake",
        fixed_standing=True,
        control_mode="arms_only",
    )
    try:
        env.reset(seed=42)
        h0_qpos = env.id_cache.joint_qpos_idx["h0"]
        h1_qpos = env.id_cache.joint_qpos_idx["h1"]
        h0_x = float(env.data.qpos[h0_qpos[0]])
        h1_x = float(env.data.qpos[h1_qpos[0]])
        dist = abs(h1_x - h0_x)
        # Should spawn near arm reach for fixed-standing handshake.
        assert 0.6 <= dist <= 1.0
    finally:
        env.close()


def test_hug_fixed_standing_spawn_is_reachable_early_stage():
    env = HumanoidCollabEnv(
        task="hug",
        stage=0,
        fixed_standing=True,
        control_mode="arms_only",
    )
    try:
        env.reset(seed=42)
        h0_qpos = env.id_cache.joint_qpos_idx["h0"]
        h1_qpos = env.id_cache.joint_qpos_idx["h1"]
        h0_x = float(env.data.qpos[h0_qpos[0]])
        h1_x = float(env.data.qpos[h1_qpos[0]])
        dist = abs(h1_x - h0_x)
        # Early fixed-standing hug stages should leave extra space to avoid foot overlap.
        assert 0.35 <= dist <= 0.55
    finally:
        env.close()


def test_hug_fixed_standing_spawn_is_reachable_late_stage():
    env = HumanoidCollabEnv(
        task="hug",
        stage=3,
        fixed_standing=True,
        control_mode="arms_only",
    )
    try:
        env.reset(seed=42)
        h0_qpos = env.id_cache.joint_qpos_idx["h0"]
        h1_qpos = env.id_cache.joint_qpos_idx["h1"]
        h0_x = float(env.data.qpos[h0_qpos[0]])
        h1_x = float(env.data.qpos[h1_qpos[0]])
        dist = abs(h1_x - h0_x)
        # Later stages should spawn closer so wrap/contact rewards are reachable.
        assert 0.15 <= dist <= 0.35
    finally:
        env.close()


def test_hug_fixed_standing_uses_lower_body_welds():
    env = HumanoidCollabEnv(task="hug", fixed_standing=True, control_mode="arms_only")
    try:
        body_by_id = {
            i: mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(env.model.nbody)
        }
        welded_bodies = set()
        for eq_idx in range(env.model.neq):
            obj1 = int(env.model.eq_obj1id[eq_idx])
            obj2 = int(env.model.eq_obj2id[eq_idx])
            n1 = body_by_id.get(obj1)
            n2 = body_by_id.get(obj2)
            if n1 is not None:
                welded_bodies.add(n1)
            if n2 is not None:
                welded_bodies.add(n2)
        assert "h0_lower_body" in welded_bodies
        assert "h1_lower_body" in welded_bodies
    finally:
        env.close()


def test_hug_arms_only_includes_abdomen_when_fixed_standing():
    env = HumanoidCollabEnv(task="hug", fixed_standing=True, control_mode="arms_only")
    try:
        # 6 arm actuators + 2 abdomen actuators
        assert env.action_space("h0").shape[0] == 8
        h0_names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
            for i in env._control_actuator_idx["h0"]
        ]
        assert any(name is not None and "_abdomen_y" in name for name in h0_names)
        assert any(name is not None and "_abdomen_z" in name for name in h0_names)
    finally:
        env.close()
