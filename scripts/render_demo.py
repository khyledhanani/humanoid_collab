"""Render a demo of any task with random actions."""

import argparse

from humanoid_collab.env import HumanoidCollabEnv
from humanoid_collab.mjx_env import MJXHumanoidCollabEnv
from humanoid_collab.tasks.registry import available_tasks


def _make_env(task: str, stage: int, backend: str, render_mode: str):
    if backend == "mjx":
        return MJXHumanoidCollabEnv(task=task, render_mode=render_mode, stage=stage)
    return HumanoidCollabEnv(task=task, render_mode=render_mode, stage=stage)


def render_human(task: str, stage: int = 0, max_steps: int = 500, backend: str = "cpu"):
    """Render with interactive viewer (requires mjpython on macOS).

    Auto-resets on episode termination so you can watch continuously.
    """
    env = _make_env(task, stage, backend, "human")
    obs, infos = env.reset(seed=42)

    print(f"Rendering task '{task}' (stage {stage}, backend={backend}). Close the viewer to stop.")
    print(f"Auto-resets on fall/termination. Will run for {max_steps} total steps.")

    step = 0
    episode = 0
    while step < max_steps:
        if not env.agents:
            episode += 1
            obs, infos = env.reset(seed=42 + episode)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        step += 1

    env.close()
    print(f"Finished after {step} steps, {episode + 1} episodes.")


def render_video(
    task: str,
    stage: int = 0,
    max_steps: int = 300,
    output: str = "demo.mp4",
    backend: str = "cpu",
):
    """Save a video of the demo."""
    try:
        import imageio
    except ImportError:
        print("imageio required for video mode: pip install imageio imageio-ffmpeg")
        return

    env = _make_env(task, stage, backend, "rgb_array")
    obs, infos = env.reset(seed=42)

    frames = []
    step = 0
    episode = 0
    while step < max_steps:
        if not env.agents:
            episode += 1
            obs, infos = env.reset(seed=42 + episode)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        step += 1

    env.close()

    if frames:
        imageio.mimsave(output, frames, fps=40)
        print(f"Saved {len(frames)} frames to {output}")
    else:
        print("No frames captured.")


def main():
    parser = argparse.ArgumentParser(description="Render demo")
    parser.add_argument("--task", type=str, default="hug",
                        choices=available_tasks(),
                        help="Task to render")
    parser.add_argument("--backend", type=str, default="cpu",
                        choices=["cpu", "mjx"],
                        help="Physics backend")
    parser.add_argument("--mode", type=str, default="video",
                        choices=["human", "video"],
                        help="Rendering mode")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=300, help="Max steps")
    parser.add_argument("--output", type=str, default="demo.mp4", help="Output video file")
    args = parser.parse_args()

    if args.mode == "human":
        render_human(args.task, args.stage, args.steps, args.backend)
    else:
        render_video(args.task, args.stage, args.steps, args.output, args.backend)


if __name__ == "__main__":
    main()
