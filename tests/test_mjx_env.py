"""Tests for MJX backend environment."""

import importlib.util

import pytest

from humanoid_collab.mjx_env import MJXHumanoidCollabEnv


def _has_jax():
    return importlib.util.find_spec("jax") is not None


def _has_mjx():
    # Either bundled under mujoco or standalone package.
    if importlib.util.find_spec("mujoco_mjx") is not None:
        return True
    try:
        import mujoco  # type: ignore
        return hasattr(mujoco, "mjx")
    except Exception:
        return False


def _has_full_mjx_stack():
    return _has_jax() and _has_mjx()


def test_mjx_env_dependency_error_is_explicit():
    if _has_full_mjx_stack():
        pytest.skip("MJX dependencies available; error path not applicable.")

    with pytest.raises(ImportError) as exc_info:
        MJXHumanoidCollabEnv(task="hug")
    msg = str(exc_info.value).lower()
    assert "mjx" in msg or "jax" in msg


@pytest.mark.skipif(not _has_full_mjx_stack(), reason="MJX stack is not installed.")
def test_mjx_env_api_smoke():
    env = MJXHumanoidCollabEnv(task="hug", horizon=50)
    obs, infos = env.reset(seed=123)

    assert "h0" in obs and "h1" in obs
    assert infos["h0"]["backend"] == "mjx"
    assert infos["h1"]["backend"] == "mjx"

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert "h0" in obs and "h1" in obs
    assert isinstance(rewards["h0"], float)
    assert rewards["h0"] == rewards["h1"]
    assert infos["h0"]["backend"] == "mjx"
    env.close()

