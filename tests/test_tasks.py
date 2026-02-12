"""Task-specific tests for success conditions, rewards, and curriculum."""

import pytest
import numpy as np
from humanoid_collab.env import HumanoidCollabEnv
from humanoid_collab.tasks.registry import get_task, available_tasks


class TestTaskRegistry:
    def test_all_tasks_available(self):
        tasks = available_tasks()
        assert "hug" in tasks
        assert "handshake" in tasks
        assert "box_lift" in tasks

    def test_get_task_returns_config(self):
        for name in available_tasks():
            config = get_task(name)
            assert config.name == name

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")


class TestHugTask:
    def test_hug_obs_dim(self):
        config = get_task("hug")
        assert config.task_obs_dim == 0

    def test_hug_curriculum_stages(self):
        config = get_task("hug")
        assert config.num_curriculum_stages == 4
        for stage in range(4):
            config.set_stage(stage)
            weights = config.get_weights_dict()
            assert isinstance(weights, dict)

    def test_hug_reward_computation(self):
        env = HumanoidCollabEnv(task="hug")
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, _, _, infos = env.step(actions)
        for agent in ["h0", "h1"]:
            assert isinstance(rewards[agent], float)
            assert "total_reward" in infos[agent]
        env.close()


class TestHandshakeTask:
    def test_handshake_obs_dim(self):
        config = get_task("handshake")
        assert config.task_obs_dim == 12

    def test_handshake_curriculum_stages(self):
        config = get_task("handshake")
        assert config.num_curriculum_stages == 4
        for stage in range(4):
            config.set_stage(stage)
            weights = config.get_weights_dict()
            assert isinstance(weights, dict)

    def test_handshake_reward_computation(self):
        env = HumanoidCollabEnv(task="handshake")
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, _, _, infos = env.step(actions)
        for agent in ["h0", "h1"]:
            assert isinstance(rewards[agent], float)
        env.close()


class TestBoxLiftTask:
    def test_box_lift_obs_dim(self):
        config = get_task("box_lift")
        assert config.task_obs_dim == 0

    def test_box_lift_curriculum_stages(self):
        config = get_task("box_lift")
        assert config.num_curriculum_stages == 4
        for stage in range(4):
            config.set_stage(stage)
            weights = config.get_weights_dict()
            assert isinstance(weights, dict)

    def test_box_lift_has_box_in_scene(self):
        env = HumanoidCollabEnv(task="box_lift")
        env.reset(seed=42)
        # Box should exist in the model
        box_pos = env.id_cache.get_site_xpos(env.data, "box_center")
        assert box_pos is not None
        assert len(box_pos) == 3
        env.close()

    def test_box_lift_box_height_in_obs(self):
        env = HumanoidCollabEnv(task="box_lift")
        obs, _ = env.reset(seed=42)
        expected_dim = env.obs_builder.get_base_obs_dim() + env.task_config.task_obs_dim
        assert obs["h0"].shape[0] == expected_dim
        env.close()

    def test_box_lift_reward_computation(self):
        env = HumanoidCollabEnv(task="box_lift")
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, _, _, infos = env.step(actions)
        for agent in ["h0", "h1"]:
            assert isinstance(rewards[agent], float)
            assert "box_height" in infos[agent]
        env.close()


class TestAllTasksReward:
    @pytest.mark.parametrize("task", ["hug", "handshake", "box_lift"])
    def test_reward_no_nan(self, task):
        env = HumanoidCollabEnv(task=task)
        env.reset(seed=42)
        for _ in range(20):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            for agent in ["h0", "h1"]:
                assert not np.isnan(rewards[agent]), f"NaN reward in {task}"
        env.close()

    @pytest.mark.parametrize("task", ["hug", "handshake", "box_lift"])
    def test_cooperative_reward(self, task):
        """Both agents should receive the same reward (cooperative)."""
        env = HumanoidCollabEnv(task=task)
        env.reset(seed=42)
        for _ in range(10):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            assert rewards["h0"] == rewards["h1"], \
                f"Rewards not equal in {task}: h0={rewards['h0']}, h1={rewards['h1']}"
        env.close()
