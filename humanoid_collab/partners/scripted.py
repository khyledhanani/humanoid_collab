"""Scripted reactive partner for Stage B SAC training.

H1's lower body is welded to the world (it cannot fall).  Its 6 arm actuators
respond to H0's proximity with a smooth, rule-based policy:

  far   (d > open_thresh)                 → IDLE    (arms at sides)
  mid   (embrace_thresh < d ≤ open_thresh) → OPEN    (arms extended forward)
  close (d ≤ embrace_thresh)              → EMBRACE (arms wrapping forward)

All transitions are smoothed so arm motion is continuous rather than snapping.

Usage in train_sac.py (Stage B):
    arm_positions = _build_arm_actuator_positions(env, partner="h1")
    partner = ScriptedPartner(
        id_cache=env.id_cache,
        arm_actuator_positions=arm_positions,
        act_dim=env._action_dim,
    )
    # Each step:
    action_h1 = partner.get_action(env.data)
    # Each episode reset:
    partner.reset()
"""
from __future__ import annotations

from typing import List

import mujoco
import numpy as np


class ScriptedPartner:
    """Rule-based arm controller for the passive partner in Stage B.

    Args:
        id_cache:               IDCache from the env (used for site lookups).
        arm_actuator_positions: Indices within the partner's action vector
                                that correspond to the 6 arm actuators, in
                                the order they appear in the MuJoCo actuator
                                list: [ls1, ls2, le, rs1, rs2, re].
        act_dim:                Full action dimension for the partner agent.
        active_agent:           Label of the RL-trained agent ("h0").
        partner_agent:          Label of the scripted partner ("h1").
        open_thresh:            Distance (m) at which arms begin to open.
        embrace_thresh:         Distance (m) at which arms shift to embrace pose.
        transition_speed:       Fraction of remaining gap closed per step
                                (higher = faster transitions).
    """

    # Normalized actuator targets per state.
    # Order: left_shoulder1, left_shoulder2, left_elbow,
    #        right_shoulder1, right_shoulder2, right_elbow
    _IDLE     = np.array([ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00], dtype=np.float32)
    _OPEN     = np.array([ 0.65, -0.35,  0.20,  0.65,  0.35,  0.20], dtype=np.float32)
    _EMBRACE  = np.array([-0.45,  0.55, -0.25, -0.45, -0.55, -0.25], dtype=np.float32)

    def __init__(
        self,
        id_cache,
        arm_actuator_positions: List[int],
        act_dim: int,
        active_agent: str = "h0",
        partner_agent: str = "h1",
        open_thresh: float = 1.5,
        embrace_thresh: float = 0.7,
        transition_speed: float = 0.05,
    ) -> None:
        self.id_cache = id_cache
        self.arm_positions = list(arm_actuator_positions)
        self.act_dim = int(act_dim)
        self.active_agent = str(active_agent)
        self.partner_agent = str(partner_agent)
        self.open_thresh = float(open_thresh)
        self.embrace_thresh = float(embrace_thresh)
        self.transition_speed = float(transition_speed)

        # Current smoothed arm state (6D, matches _IDLE / _OPEN / _EMBRACE)
        self._current = self._IDLE.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset arm state to idle at episode start."""
        self._current = self._IDLE.copy()

    def get_action(self, data: mujoco.MjData) -> np.ndarray:
        """Compute the scripted partner's full action vector.

        Args:
            data: Current MuJoCo simulation data.

        Returns:
            Float32 array of shape (act_dim,) clipped to [-1, 1].
            Leg/abdomen indices are zero; arm indices carry the scripted targets.
        """
        chest_active  = self.id_cache.get_site_xpos(data, f"{self.active_agent}_chest")
        chest_partner = self.id_cache.get_site_xpos(data, f"{self.partner_agent}_chest")
        dist = float(np.linalg.norm(chest_active - chest_partner))
        return self.get_action_from_distance(dist)

    def get_action_from_distance(self, chest_distance: float) -> np.ndarray:
        """Compute action from chest distance only.

        This is useful for vectorized subprocess training where callers do not
        have direct access to per-env MuJoCo `data` objects.
        """
        dist = float(chest_distance)

        # Select target pose
        if dist <= self.embrace_thresh:
            target = self._EMBRACE
        elif dist <= self.open_thresh:
            # Linear interpolation: 0 = embrace, 1 = open
            t = (dist - self.embrace_thresh) / (self.open_thresh - self.embrace_thresh)
            target = (1.0 - t) * self._EMBRACE + t * self._OPEN
        else:
            target = self._IDLE

        # Smooth exponential tracking toward target
        self._current += self.transition_speed * (target - self._current)

        # Build full action vector (zeros for non-arm actuators)
        action = np.zeros(self.act_dim, dtype=np.float32)
        for i, pos in enumerate(self.arm_positions):
            if i < 6:
                action[pos] = float(self._current[i])

        return np.clip(action, -1.0, 1.0)
