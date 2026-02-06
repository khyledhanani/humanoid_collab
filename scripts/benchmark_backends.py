"""Benchmark CPU backend throughput for HumanoidCollab.

Modes:
- single: Python step loop over one env
- subproc: subprocess vectorized CPU envs
- both: run both and print speedup
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List

import numpy as np

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles
from humanoid_collab.vector_env import SubprocHumanoidCollabVecEnv


def _run_steps_single(
    env: HumanoidCollabEnv,
    steps: int,
    rng: np.random.RandomState,
    act_dim: int,
    seed_base: int,
) -> float:
    t0 = time.perf_counter()
    for i in range(steps):
        if not env.agents:
            env.reset(seed=seed_base + i + 1)
        action_h0 = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        action_h1 = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        env.step({"h0": action_h0, "h1": action_h1})
    return time.perf_counter() - t0


def _benchmark_single(
    task: str,
    frame_skip: int,
    physics_profile: str,
    fixed_standing: bool,
    control_mode: str,
    horizons: List[int],
    repeats: int,
    seed: int,
) -> Dict[str, object]:
    env = HumanoidCollabEnv(
        task=task,
        horizon=max(horizons) + 1000,
        frame_skip=frame_skip,
        physics_profile=physics_profile,
        fixed_standing=fixed_standing,
        control_mode=control_mode,
    )
    env.reset(seed=seed)
    act_dim = int(env.action_space("h0").shape[0])

    horizon_times: Dict[int, List[float]] = {h: [] for h in horizons}
    for repeat in range(repeats):
        rng = np.random.RandomState(seed + repeat + 1)
        for horizon in horizons:
            env.reset(seed=seed + repeat + horizon)
            dt = _run_steps_single(env, horizon, rng, act_dim, seed + repeat + horizon * 10)
            horizon_times[horizon].append(dt)

    env.close()

    horizon_summary = {}
    for horizon, dts in horizon_times.items():
        mean_s = statistics.fmean(dts)
        stdev_s = statistics.pstdev(dts) if len(dts) > 1 else 0.0
        effective_steps = float(horizon)
        horizon_summary[horizon] = {
            "mean_s": mean_s,
            "stdev_s": stdev_s,
            "steps_per_s": effective_steps / mean_s if mean_s > 0 else 0.0,
            "effective_steps": effective_steps,
        }

    return {
        "backend": "cpu-single",
        "horizons": horizon_summary,
        "effective_multiplier": 1.0,
    }


def _run_steps_subproc(
    vec_env: SubprocHumanoidCollabVecEnv,
    steps: int,
    rng: np.random.RandomState,
    act_dim: int,
) -> float:
    t0 = time.perf_counter()
    num_envs = vec_env.num_envs
    for _ in range(steps):
        actions = {
            "h0": rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32),
            "h1": rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32),
        }
        vec_env.step(actions)
    return time.perf_counter() - t0


def _benchmark_subproc(
    task: str,
    frame_skip: int,
    physics_profile: str,
    fixed_standing: bool,
    control_mode: str,
    num_envs: int,
    horizons: List[int],
    repeats: int,
    seed: int,
    start_method: str,
) -> Dict[str, object]:
    env_kwargs = {
        "task": task,
        "horizon": max(horizons) + 1000,
        "frame_skip": frame_skip,
        "physics_profile": physics_profile,
        "fixed_standing": fixed_standing,
        "control_mode": control_mode,
    }
    vec_env = SubprocHumanoidCollabVecEnv(
        num_envs=num_envs,
        env_kwargs=env_kwargs,
        auto_reset=True,
        start_method=start_method,
    )
    vec_env.reset(seed=seed)
    act_dim = int(vec_env.action_space("h0").shape[0])

    horizon_times: Dict[int, List[float]] = {h: [] for h in horizons}
    for repeat in range(repeats):
        rng = np.random.RandomState(seed + repeat + 1)
        for horizon in horizons:
            vec_env.reset(seed=seed + repeat + horizon)
            dt = _run_steps_subproc(vec_env, horizon, rng, act_dim)
            horizon_times[horizon].append(dt)

    vec_env.close()

    horizon_summary = {}
    multiplier = float(num_envs)
    for horizon, dts in horizon_times.items():
        mean_s = statistics.fmean(dts)
        stdev_s = statistics.pstdev(dts) if len(dts) > 1 else 0.0
        effective_steps = float(horizon) * multiplier
        horizon_summary[horizon] = {
            "mean_s": mean_s,
            "stdev_s": stdev_s,
            "steps_per_s": effective_steps / mean_s if mean_s > 0 else 0.0,
            "effective_steps": effective_steps,
        }

    return {
        "backend": "cpu-subproc",
        "horizons": horizon_summary,
        "effective_multiplier": multiplier,
        "num_envs": num_envs,
        "start_method": start_method,
    }


def _print_results(results: Dict[str, object]) -> None:
    backend = str(results["backend"])
    horizons: Dict[int, Dict[str, float]] = results["horizons"]  # type: ignore[assignment]
    num_envs = int(results.get("num_envs", 1))
    start_method = str(results.get("start_method", ""))

    print(f"\n[{backend.upper()}]")
    if num_envs > 1:
        print(f"num_envs: {num_envs}")
    if start_method:
        print(f"start_method: {start_method}")
    for h in sorted(horizons.keys()):
        row = horizons[h]
        print(
            f"steps={h:>6d} | effective_steps={int(row['effective_steps']):>8d} "
            f"| mean={row['mean_s']:.4f}s | std={row['stdev_s']:.4f}s "
            f"| throughput={row['steps_per_s']:.1f} steps/s"
        )


def _print_speedup(single: Dict[str, object], subproc: Dict[str, object]) -> None:
    h_single: Dict[int, Dict[str, float]] = single["horizons"]  # type: ignore[assignment]
    h_subproc: Dict[int, Dict[str, float]] = subproc["horizons"]  # type: ignore[assignment]
    common_h = sorted(set(h_single.keys()) & set(h_subproc.keys()))
    if not common_h:
        return
    print("\n[SPEEDUP: SUBPROC VS SINGLE]")
    for h in common_h:
        t_single = float(h_single[h]["steps_per_s"])
        t_subproc = float(h_subproc[h]["steps_per_s"])
        ratio = (t_subproc / t_single) if t_single > 0 else 0.0
        print(f"horizon={h:>6d} | single={t_single:>10.1f} | subproc={t_subproc:>10.1f} | x{ratio:.2f}")


def parse_args() -> argparse.Namespace:
    physics_profiles = available_physics_profiles()

    parser = argparse.ArgumentParser(description="Benchmark HumanoidCollab CPU backend.")
    parser.add_argument("--task", type=str, default="hug", choices=["hug", "handshake", "box_lift"])
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["single", "subproc", "both"],
        help="Which CPU benchmark mode(s) to run.",
    )
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="default",
        choices=physics_profiles,
        help="Physics profile for CPU backend.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of subprocess envs for subproc mode.",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default="forkserver",
        choices=["fork", "forkserver", "spawn"],
        help="Multiprocessing start method for subproc mode.",
    )
    parser.add_argument(
        "--fixed-standing",
        action="store_true",
        help="Weld torsos to world (disable locomotion).",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="all",
        choices=["all", "arms_only"],
        help="Actuator subset controlled by RL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))

    run_single = args.mode in {"single", "both"}
    run_subproc = args.mode in {"subproc", "both"}

    single_results = None
    subproc_results = None

    if run_single:
        single_results = _benchmark_single(
            task=args.task,
            frame_skip=args.frame_skip,
            physics_profile=args.physics_profile,
            fixed_standing=args.fixed_standing,
            control_mode=args.control_mode,
            horizons=horizons,
            repeats=args.repeats,
            seed=args.seed,
        )
        _print_results(single_results)

    if run_subproc:
        if args.num_envs <= 0:
            raise ValueError("--num-envs must be > 0")
        subproc_results = _benchmark_subproc(
            task=args.task,
            frame_skip=args.frame_skip,
            physics_profile=args.physics_profile,
            fixed_standing=args.fixed_standing,
            control_mode=args.control_mode,
            num_envs=args.num_envs,
            horizons=horizons,
            repeats=args.repeats,
            seed=args.seed,
            start_method=args.start_method,
        )
        _print_results(subproc_results)

    if single_results is not None and subproc_results is not None:
        _print_speedup(single_results, subproc_results)


if __name__ == "__main__":
    main()
