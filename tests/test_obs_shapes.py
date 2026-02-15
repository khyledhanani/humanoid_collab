"""Tests for observation and action shape consistency across all tasks."""

import pytest
import numpy as np
from humanoid_collab.env import HumanoidCollabEnv


TASKS = ["hug", "handshake", "box_lift", "walk_to_target"]


@pytest.fixture(params=TASKS)
def env(request):
    e = HumanoidCollabEnv(task=request.param)
    yield e
    e.close()


class TestObsShape:
    def test_base_obs_is_self_proprio_only(self, env):
        h0_qpos = len(env.id_cache.joint_qpos_idx["h0"])
        h0_qvel = len(env.id_cache.joint_qvel_idx["h0"])
        expected_base = (h0_qpos - 7) + (h0_qvel - 6) + 4 + 3
        assert env.obs_builder.get_base_obs_dim() == expected_base
        assert env.obs_builder.get_obs_dim() == expected_base + env.task_config.task_obs_dim

    def test_obs_matches_declared_space(self, env):
        obs, _ = env.reset(seed=42)
        for agent in ["h0", "h1"]:
            expected_shape = env.observation_space(agent).shape
            assert obs[agent].shape == expected_shape, \
                f"Obs shape {obs[agent].shape} != space shape {expected_shape}"

    def test_obs_dtype_float32(self, env):
        obs, _ = env.reset(seed=42)
        for agent in ["h0", "h1"]:
            assert obs[agent].dtype == np.float32

    def test_obs_shape_consistent_across_steps(self, env):
        obs, _ = env.reset(seed=42)
        initial_shapes = {agent: obs[agent].shape for agent in ["h0", "h1"]}

        for _ in range(50):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)
            for agent in ["h0", "h1"]:
                assert obs[agent].shape == initial_shapes[agent]

    def test_no_nan_in_obs(self, env):
        obs, _ = env.reset(seed=42)
        for agent in ["h0", "h1"]:
            assert not np.any(np.isnan(obs[agent])), "NaN in reset observations"

        for _ in range(50):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)
            for agent in ["h0", "h1"]:
                assert not np.any(np.isnan(obs[agent])), "NaN in step observations"

    def test_no_inf_in_obs(self, env):
        obs, _ = env.reset(seed=42)
        for agent in ["h0", "h1"]:
            assert not np.any(np.isinf(obs[agent]))

    def test_obs_values_reasonable(self, env):
        obs, _ = env.reset(seed=42)
        for agent in ["h0", "h1"]:
            assert np.all(np.abs(obs[agent]) < 1000), "Obs values unreasonably large"


class TestActionShape:
    def test_action_dim_matches_actuators(self, env):
        action_dim = env.action_space("h0").shape[0]
        assert action_dim == 22, f"Expected 22 actuators, got {action_dim}"

    def test_action_bounds(self, env):
        space = env.action_space("h0")
        assert np.all(space.low == -1.0)
        assert np.all(space.high == 1.0)

    def test_h0_h1_same_action_space(self, env):
        s0 = env.action_space("h0")
        s1 = env.action_space("h1")
        assert s0.shape == s1.shape
        np.testing.assert_array_equal(s0.low, s1.low)
        np.testing.assert_array_equal(s0.high, s1.high)

    def test_h0_h1_same_obs_space(self, env):
        s0 = env.observation_space("h0")
        s1 = env.observation_space("h1")
        assert s0.shape == s1.shape
