"""Tests for fixed-standing and arms-only control modes."""

import numpy as np

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

