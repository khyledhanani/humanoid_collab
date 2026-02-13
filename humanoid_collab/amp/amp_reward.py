"""AMP reward computation for natural motion style."""

from typing import Dict, Optional
import numpy as np
import torch
import mujoco

from humanoid_collab.utils.ids import IDCache
from humanoid_collab.amp.amp_obs import AMPObsBuilder
from humanoid_collab.amp.discriminator import AMPDiscriminator


class AMPRewardComputer:
    """Computes AMP style reward from discriminator.

    Maintains previous observations to compute state transitions
    and provides reward for natural motion.
    """

    def __init__(
        self,
        discriminator: AMPDiscriminator,
        amp_obs_builder: AMPObsBuilder,
        reward_scale: float = 2.0,
        clip_reward: bool = True,
        device: str = "cpu",
    ):
        """Initialize the AMP reward computer.

        Args:
            discriminator: Trained discriminator network
            amp_obs_builder: Builder for AMP observations
            reward_scale: Scale factor for style reward
            clip_reward: Use clipped stable reward form
            device: Device for discriminator inference
        """
        self.discriminator = discriminator.to(device)
        self.discriminator.eval()
        self.amp_obs_builder = amp_obs_builder
        self.reward_scale = reward_scale
        self.clip_reward = clip_reward
        self.device = device

        # Cache previous observations for each agent
        self._prev_obs: Dict[str, np.ndarray] = {}
        self._initialized = False

    def reset(self, data: mujoco.MjData, id_cache: IDCache) -> None:
        """Reset and cache initial AMP observations.

        Args:
            data: MuJoCo data instance
            id_cache: Cached IDs from the MuJoCo model
        """
        self._prev_obs = {}
        for agent in ["h0", "h1"]:
            self._prev_obs[agent] = self.amp_obs_builder.compute_obs(data, agent)
        self._initialized = True

    def compute_reward(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
    ) -> Dict[str, float]:
        """Compute AMP style reward for both agents.

        Args:
            data: MuJoCo data instance
            id_cache: Cached IDs from the MuJoCo model

        Returns:
            Dictionary mapping agent ID to style reward
        """
        if not self._initialized:
            raise RuntimeError("AMPRewardComputer not initialized. Call reset() first.")

        rewards = {}

        for agent in ["h0", "h1"]:
            # Compute current observation
            curr_obs = self.amp_obs_builder.compute_obs(data, agent)

            # Compute reward from discriminator
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    self._prev_obs[agent], device=self.device
                ).unsqueeze(0)
                obs_t1 = torch.as_tensor(curr_obs, device=self.device).unsqueeze(0)

                reward = self.discriminator.compute_reward(
                    obs_t, obs_t1,
                    clip_reward=self.clip_reward,
                    reward_scale=self.reward_scale,
                )
                rewards[agent] = reward.item()

            # Update cached observation
            self._prev_obs[agent] = curr_obs

        return rewards

    def compute_reward_batch(
        self,
        obs_t: np.ndarray,
        obs_t1: np.ndarray,
    ) -> np.ndarray:
        """Compute AMP reward for a batch of transitions.

        Args:
            obs_t: AMP observations at time t (batch, obs_dim)
            obs_t1: AMP observations at time t+1 (batch, obs_dim)

        Returns:
            Style rewards (batch,)
        """
        with torch.no_grad():
            obs_t_tensor = torch.as_tensor(obs_t, device=self.device)
            obs_t1_tensor = torch.as_tensor(obs_t1, device=self.device)

            reward = self.discriminator.compute_reward(
                obs_t_tensor, obs_t1_tensor,
                clip_reward=self.clip_reward,
                reward_scale=self.reward_scale,
            )
            return reward.cpu().numpy()

    def get_current_obs(self, agent: str) -> Optional[np.ndarray]:
        """Get the cached observation for an agent.

        Args:
            agent: Agent ID

        Returns:
            Cached observation or None if not initialized
        """
        return self._prev_obs.get(agent)

    def get_transition_obs(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        agent: str,
    ) -> tuple:
        """Get the transition observations for an agent without updating cache.

        Args:
            data: MuJoCo data instance
            id_cache: Cached IDs
            agent: Agent ID

        Returns:
            Tuple of (obs_t, obs_t1)
        """
        if agent not in self._prev_obs:
            raise RuntimeError(f"No cached observation for agent {agent}")

        curr_obs = self.amp_obs_builder.compute_obs(data, agent)
        return self._prev_obs[agent].copy(), curr_obs


class CombinedRewardComputer:
    """Combines AMP style reward with task reward.

    Manages the blending of AMP and task rewards according to
    a weight schedule.
    """

    def __init__(
        self,
        amp_reward_computer: AMPRewardComputer,
        amp_weight: float = 0.5,
        task_weight: float = 0.5,
        alive_bonus: float = 0.1,
        energy_penalty: float = 0.0005,
        fall_penalty: float = 10.0,
    ):
        """Initialize the combined reward computer.

        Args:
            amp_reward_computer: AMP reward computer
            amp_weight: Weight for AMP reward
            task_weight: Weight for task reward
            alive_bonus: Bonus for staying upright
            energy_penalty: Penalty coefficient for control effort
            fall_penalty: Penalty for falling
        """
        self.amp_reward_computer = amp_reward_computer
        self.amp_weight = amp_weight
        self.task_weight = task_weight
        self.alive_bonus = alive_bonus
        self.energy_penalty = energy_penalty
        self.fall_penalty = fall_penalty

    def set_weights(self, amp_weight: float, task_weight: float) -> None:
        """Update the reward weights.

        Args:
            amp_weight: New AMP weight
            task_weight: New task weight
        """
        self.amp_weight = amp_weight
        self.task_weight = task_weight

    def compute_reward(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        task_reward: float,
        ctrl: np.ndarray,
        fallen: bool,
    ) -> Dict[str, float]:
        """Compute combined reward for both agents.

        Args:
            data: MuJoCo data instance
            id_cache: Cached IDs
            task_reward: Task-specific reward (same for both agents)
            ctrl: Control actions
            fallen: Whether either agent has fallen

        Returns:
            Dictionary mapping agent ID to combined reward
        """
        # Compute AMP rewards
        amp_rewards = self.amp_reward_computer.compute_reward(data, id_cache)

        # Compute combined reward for each agent
        combined = {}

        for agent in ["h0", "h1"]:
            r_amp = amp_rewards[agent]
            r_task = task_reward

            # Weighted combination
            r_combined = self.amp_weight * r_amp + self.task_weight * r_task

            # Alive bonus
            r_combined += self.alive_bonus

            # Energy penalty
            r_combined -= self.energy_penalty * np.sum(ctrl ** 2)

            # Fall penalty
            if fallen:
                r_combined -= self.fall_penalty

            combined[agent] = r_combined

        return combined

    def compute_pretrain_reward(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        ctrl: np.ndarray,
        fallen: bool,
    ) -> Dict[str, float]:
        """Compute pre-training reward (AMP only + alive bonus).

        Args:
            data: MuJoCo data instance
            id_cache: Cached IDs
            ctrl: Control actions
            fallen: Whether either agent has fallen

        Returns:
            Dictionary mapping agent ID to pre-training reward
        """
        # Compute AMP rewards
        amp_rewards = self.amp_reward_computer.compute_reward(data, id_cache)

        rewards = {}

        for agent in ["h0", "h1"]:
            r = amp_rewards[agent]

            # Alive bonus
            r += self.alive_bonus

            # Energy penalty
            r -= self.energy_penalty * np.sum(ctrl ** 2)

            # Fall penalty
            if fallen:
                r -= self.fall_penalty

            rewards[agent] = r

        return rewards
