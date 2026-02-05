"""Tests for contact detection logic across all tasks."""

import pytest
import numpy as np
from humanoid_collab.env import HumanoidCollabEnv


TASKS = ["hug", "handshake", "box_lift"]


@pytest.fixture(params=TASKS)
def env(request):
    e = HumanoidCollabEnv(task=request.param)
    yield e
    e.close()


class TestContactDetection:
    def test_contact_detector_initializes(self, env):
        assert env.contact_detector is not None

    def test_detect_contacts_returns_dict(self, env):
        env.reset(seed=42)
        contact_info = env.contact_detector.detect_contacts(env.data)
        assert isinstance(contact_info, dict)

    def test_contact_values_are_booleans(self, env):
        env.reset(seed=42)
        contact_info = env.contact_detector.detect_contacts(env.data)
        for key, val in contact_info.items():
            assert isinstance(val, bool), f"Contact '{key}' is {type(val)}, expected bool"

    def test_contacts_run_multiple_steps(self, env):
        env.reset(seed=42)
        for _ in range(50):
            if not env.agents:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
            contact_info = env.contact_detector.detect_contacts(env.data)
            assert isinstance(contact_info, dict)

    def test_contact_force_proxy(self, env):
        env.reset(seed=42)
        force = env.contact_detector.get_contact_force_proxy(env.data)
        assert isinstance(force, float)
        assert force >= 0.0


class TestHugContactKeys:
    def test_hug_contact_keys(self):
        env = HumanoidCollabEnv(task="hug")
        env.reset(seed=42)
        contact_info = env.contact_detector.detect_contacts(env.data)
        expected_keys = [
            "h0_arm_h1_torso", "h1_arm_h0_torso",
            "h0_l_arm_h1_torso", "h0_r_arm_h1_torso",
            "h1_l_arm_h0_torso", "h1_r_arm_h0_torso",
        ]
        for key in expected_keys:
            assert key in contact_info, f"Missing key '{key}' in hug contacts"
        env.close()


class TestHandshakeContactKeys:
    def test_handshake_contact_keys(self):
        env = HumanoidCollabEnv(task="handshake")
        env.reset(seed=42)
        contact_info = env.contact_detector.detect_contacts(env.data)
        expected_keys = [
            "h0_hand_h1_hand",
            "h0_r_hand_h1_r_hand",
        ]
        for key in expected_keys:
            assert key in contact_info, f"Missing key '{key}' in handshake contacts"
        env.close()


class TestBoxLiftContactKeys:
    def test_box_lift_contact_keys(self):
        env = HumanoidCollabEnv(task="box_lift")
        env.reset(seed=42)
        contact_info = env.contact_detector.detect_contacts(env.data)
        expected_keys = [
            "h0_hand_box", "h1_hand_box",
            "h0_l_hand_box", "h0_r_hand_box",
            "h1_l_hand_box", "h1_r_hand_box",
        ]
        for key in expected_keys:
            assert key in contact_info, f"Missing key '{key}' in box_lift contacts"
        env.close()


class TestGeomGroups:
    def test_standard_geom_groups_exist(self, env):
        for agent in ["h0", "h1"]:
            for group in ["arm", "torso", "l_arm", "r_arm", "hand", "l_hand", "r_hand"]:
                name = f"{agent}_{group}"
                geoms = env.id_cache.get_geom_group(name)
                assert isinstance(geoms, set), f"Geom group '{name}' not a set"

    def test_arm_geoms_nonempty(self, env):
        for agent in ["h0", "h1"]:
            assert len(env.id_cache.arm_geoms[agent]) > 0

    def test_torso_geoms_nonempty(self, env):
        for agent in ["h0", "h1"]:
            assert len(env.id_cache.torso_geoms[agent]) > 0

    def test_hand_geoms_nonempty(self, env):
        for agent in ["h0", "h1"]:
            assert len(env.id_cache.hand_geoms[agent]) > 0
