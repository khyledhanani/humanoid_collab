"""Tests for egocentric visual observation modes (RGB/gray)."""

import numpy as np
import pytest

from humanoid_collab.env import HumanoidCollabEnv


def _reset_or_skip(env: HumanoidCollabEnv, seed: int):
    try:
        return env.reset(seed=seed)
    except RuntimeError as exc:
        env.close()
        pytest.skip(str(exc))


def test_rgb_observation_space_shape_and_dtype():
    try:
        env = HumanoidCollabEnv(task="hug", observation_mode="rgb", obs_rgb_width=48, obs_rgb_height=32)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    try:
        space = env.observation_space("h0")
        assert space.shape == (32, 48, 3)
        assert space.dtype == np.uint8
        assert int(space.low.min()) == 0
        assert int(space.high.max()) == 255
    finally:
        env.close()


def test_rgb_reset_returns_egocentric_frames():
    try:
        env = HumanoidCollabEnv(task="handshake", observation_mode="rgb", obs_rgb_width=40, obs_rgb_height=30)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    try:
        obs, infos = _reset_or_skip(env, seed=123)
        for agent in ["h0", "h1"]:
            assert isinstance(obs[agent], np.ndarray)
            assert obs[agent].shape == (30, 40, 3)
            assert obs[agent].dtype == np.uint8
            assert infos[agent]["observation_mode"] == "rgb"
    finally:
        env.close()


def test_rgb_step_keeps_shape_and_dtype():
    try:
        env = HumanoidCollabEnv(task="box_lift", observation_mode="rgb", obs_rgb_width=36, obs_rgb_height=24)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    try:
        obs, _ = _reset_or_skip(env, seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        for agent in ["h0", "h1"]:
            assert obs[agent].shape == (24, 36, 3)
            assert obs[agent].dtype == np.uint8
    finally:
        env.close()


def test_invalid_observation_mode_raises():
    with pytest.raises(ValueError):
        HumanoidCollabEnv(task="hug", observation_mode="invalid_mode")


def test_gray_observation_space_shape_and_dtype():
    try:
        env = HumanoidCollabEnv(task="hug", observation_mode="gray", obs_rgb_width=48, obs_rgb_height=32)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    try:
        space = env.observation_space("h0")
        assert space.shape == (32, 48, 1)
        assert space.dtype == np.uint8
        assert int(space.low.min()) == 0
        assert int(space.high.max()) == 255
    finally:
        env.close()


def test_gray_reset_returns_egocentric_frames():
    try:
        env = HumanoidCollabEnv(task="handshake", observation_mode="gray", obs_rgb_width=40, obs_rgb_height=30)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    try:
        obs, infos = _reset_or_skip(env, seed=7)
        for agent in ["h0", "h1"]:
            assert isinstance(obs[agent], np.ndarray)
            assert obs[agent].shape == (30, 40, 1)
            assert obs[agent].dtype == np.uint8
            assert infos[agent]["observation_mode"] == "gray"
    finally:
        env.close()
