"""Tests for PettingZoo API conformance across all tasks."""

import pytest
import numpy as np
from humanoid_collab.env import HumanoidCollabEnv


TASKS = ["hug", "handshake", "box_lift", "walk_to_target"]


@pytest.fixture(params=TASKS)
def env(request):
    e = HumanoidCollabEnv(task=request.param)
    yield e
    e.close()


class TestReset:
    def test_reset_returns_obs_and_infos(self, env):
        obs, infos = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)
        assert "h0" in obs and "h1" in obs
        assert "h0" in infos and "h1" in infos

    def test_reset_obs_types(self, env):
        obs, _ = env.reset(seed=42)
        for agent in ["h0", "h1"]:
            assert isinstance(obs[agent], np.ndarray)
            assert obs[agent].dtype == np.float32

    def test_reset_with_seed_deterministic(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        for agent in ["h0", "h1"]:
            np.testing.assert_array_equal(obs1[agent], obs2[agent])


class TestStep:
    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        result = env.step(actions)
        assert len(result) == 5
        obs, rewards, terminations, truncations, infos = result
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

    def test_step_keys_match(self, env):
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for d in [obs, rewards, terminations, truncations, infos]:
            assert "h0" in d and "h1" in d

    def test_100_random_steps_no_crash(self, env):
        env.reset(seed=42)
        for _ in range(100):
            if not env.agents:
                env.reset(seed=42)
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)


class TestAgents:
    def test_agents_list(self, env):
        env.reset(seed=42)
        assert env.agents == ["h0", "h1"]
        assert env.possible_agents == ["h0", "h1"]

    def test_agents_cleared_on_termination(self, env):
        """Run until episode ends and verify agents list is cleared."""
        env.reset(seed=42)
        for _ in range(env.horizon + 10):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
        assert env.agents == []


class TestSpaces:
    def test_observation_space_valid(self, env):
        space = env.observation_space("h0")
        assert space.shape[0] > 0
        assert space.dtype == np.float32

    def test_action_space_valid(self, env):
        space = env.action_space("h0")
        assert space.shape[0] > 0
        assert space.dtype == np.float32
        np.testing.assert_array_equal(space.low, -1.0 * np.ones(space.shape))
        np.testing.assert_array_equal(space.high, 1.0 * np.ones(space.shape))


class TestCurriculum:
    def test_stage_option(self, env):
        obs, infos = env.reset(seed=42, options={"stage": 1})
        for agent in ["h0", "h1"]:
            assert infos[agent]["stage"] == 1

    def test_different_stages(self, env):
        for stage in range(env.task_config.num_curriculum_stages):
            obs, infos = env.reset(seed=42, options={"stage": stage})
            assert infos["h0"]["stage"] == stage


class TestHorizon:
    def test_truncation_at_horizon(self):
        env = HumanoidCollabEnv(task="hug", horizon=50)
        env.reset(seed=42)
        for i in range(60):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
        # Should have terminated or truncated by now
        assert env.agents == []
        env.close()
