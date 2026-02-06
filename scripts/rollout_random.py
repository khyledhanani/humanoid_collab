"""Run random rollouts on any task to verify the environment works."""

import argparse
import numpy as np

from humanoid_collab.env import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.mjx_env import MJXHumanoidCollabEnv
from humanoid_collab.tasks.registry import available_tasks


def run_rollout(
    task: str,
    num_episodes: int = 3,
    stage: int = 0,
    seed: int = 42,
    backend: str = "cpu",
    physics_profile: str = "default",
    fixed_standing: bool = False,
    control_mode: str = "all",
):
    if backend == "mjx":
        env = MJXHumanoidCollabEnv(
            task=task,
            horizon=1000,
            stage=stage,
            physics_profile=physics_profile,
            fixed_standing=fixed_standing,
            control_mode=control_mode,
        )
    else:
        env = HumanoidCollabEnv(
            task=task,
            horizon=1000,
            stage=stage,
            physics_profile=physics_profile,
            fixed_standing=fixed_standing,
            control_mode=control_mode,
        )

    print(f"Task: {task}")
    print(f"Stage: {stage}")
    print(f"Backend: {backend}")
    print(f"Physics profile: {physics_profile}")
    print(f"Fixed standing: {fixed_standing}")
    print(f"Control mode: {control_mode}")
    print(f"Obs dim: {env.observation_space('h0').shape}")
    print(f"Act dim: {env.action_space('h0').shape}")
    print(f"Running {num_episodes} episodes with random actions...\n")

    episode_stats = []

    for ep in range(num_episodes):
        obs, infos = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        max_hold = 0
        termination_reason = None

        while env.agents:
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += rewards.get("h0", 0.0)
            steps += 1

            # Track hold steps from info
            for agent in ["h0", "h1"]:
                if agent in infos and "hold_steps" in infos[agent]:
                    max_hold = max(max_hold, infos[agent]["hold_steps"])

            # Get termination reason
            for agent in ["h0", "h1"]:
                if agent in infos and infos[agent].get("termination_reason"):
                    termination_reason = infos[agent]["termination_reason"]

        episode_stats.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "max_hold_steps": max_hold,
            "termination_reason": termination_reason or "horizon",
        })

        print(f"Episode {ep}: reward={total_reward:.2f}, steps={steps}, "
              f"max_hold={max_hold}, reason={termination_reason or 'horizon'}")

    env.close()

    # Summary
    rewards = [s["total_reward"] for s in episode_stats]
    steps_list = [s["steps"] for s in episode_stats]
    holds = [s["max_hold_steps"] for s in episode_stats]

    print(f"\n--- Summary ---")
    print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean steps:  {np.mean(steps_list):.1f}")
    print(f"Max hold:    {max(holds)}")


def main():
    parser = argparse.ArgumentParser(description="Run random rollouts")
    parser.add_argument("--task", type=str, default="hug",
                        choices=available_tasks(),
                        help="Task to run")
    parser.add_argument("--backend", type=str, default="cpu",
                        choices=["cpu", "mjx"],
                        help="Physics backend")
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="default",
        choices=available_physics_profiles(),
        help="MuJoCo physics profile.",
    )
    parser.add_argument(
        "--fixed-standing",
        action="store_true",
        help="Weld torsos to world to disable locomotion.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="all",
        choices=["all", "arms_only"],
        help="Actuator subset controlled by RL.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_rollout(
        args.task,
        args.episodes,
        args.stage,
        args.seed,
        args.backend,
        args.physics_profile,
        args.fixed_standing,
        args.control_mode,
    )


if __name__ == "__main__":
    main()
