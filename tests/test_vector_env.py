"""Tests for subprocess vectorized CPU environment."""

import numpy as np
import pytest

from humanoid_collab.vector_env import (
    SharedMemHumanoidCollabVecEnv,
    SubprocHumanoidCollabVecEnv,
)


@pytest.mark.parametrize(
    "vec_cls",
    [SubprocHumanoidCollabVecEnv, SharedMemHumanoidCollabVecEnv],
)
def test_vector_env_reset_and_step_shapes(vec_cls):
    vec = vec_cls(
        num_envs=2,
        env_kwargs={"task": "hug", "horizon": 64, "physics_profile": "default"},
        auto_reset=True,
        start_method="spawn",
    )
    try:
        obs, infos = vec.reset(seed=123)
        assert obs["h0"].shape[0] == 2
        assert obs["h1"].shape[0] == 2
        assert len(infos) == 2

        act_dim = vec.action_space("h0").shape[0]
        actions = {
            "h0": np.zeros((2, act_dim), dtype=np.float32),
            "h1": np.zeros((2, act_dim), dtype=np.float32),
        }
        obs, rewards, terms, truncs, infos = vec.step(actions)
        assert obs["h0"].shape == obs["h1"].shape
        assert rewards["h0"].shape == (2,)
        assert terms["h0"].shape == (2,)
        assert truncs["h0"].shape == (2,)
        assert len(infos) == 2
    finally:
        vec.close()


@pytest.mark.parametrize(
    "vec_cls",
    [SubprocHumanoidCollabVecEnv, SharedMemHumanoidCollabVecEnv],
)
def test_vector_env_auto_reset_runs_past_horizon(vec_cls):
    vec = vec_cls(
        num_envs=2,
        env_kwargs={"task": "hug", "horizon": 2, "physics_profile": "default"},
        auto_reset=True,
        start_method="spawn",
    )
    try:
        vec.reset(seed=42)
        act_dim = vec.action_space("h0").shape[0]
        actions = {
            "h0": np.random.uniform(-1, 1, size=(2, act_dim)).astype(np.float32),
            "h1": np.random.uniform(-1, 1, size=(2, act_dim)).astype(np.float32),
        }

        # Should not raise even though envs terminate/truncate quickly.
        for _ in range(6):
            obs, rewards, terms, truncs, infos = vec.step(actions)
            assert obs["h0"].shape[0] == 2
    finally:
        vec.close()
