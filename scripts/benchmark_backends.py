"""Benchmark CPU vs MJX backend speed and estimate JIT break-even length.

MJX modes:
- pettingzoo: Python env.step loop (compatibility path, slowest)
- scan: on-device JAX scan over one env
- scan-batched: on-device JAX scan over batched envs (max-throughput path)
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Optional, Type

import numpy as np

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.mjcf_builder import available_physics_profiles

try:
    from humanoid_collab import MJXHumanoidCollabEnv
except Exception:
    MJXHumanoidCollabEnv = None  # type: ignore[assignment]


EnvType = Type[HumanoidCollabEnv]


def _run_steps(
    env: HumanoidCollabEnv,
    steps: int,
    rng: np.random.RandomState,
    act_dim: int,
    seed_base: int,
) -> float:
    """Run fixed number of env steps and return elapsed seconds."""
    t0 = time.perf_counter()
    for i in range(steps):
        if not env.agents:
            env.reset(seed=seed_base + i + 1)
        action_h0 = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        action_h1 = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        env.step({"h0": action_h0, "h1": action_h1})
    return time.perf_counter() - t0


def _benchmark_backend_step_loop(
    env_cls: EnvType,
    backend_name: str,
    task: str,
    frame_skip: int,
    physics_profile: str,
    horizons: List[int],
    repeats: int,
    warmup_steps: int,
    seed: int,
) -> Dict[str, object]:
    """Benchmark step-loop backend over horizons and repeats."""
    env = env_cls(
        task=task,
        horizon=max(horizons) + warmup_steps + 1000,
        frame_skip=frame_skip,
        physics_profile=physics_profile,
    )
    env.reset(seed=seed)
    act_dim = int(env.action_space("h0").shape[0])

    compile_s: Optional[float] = None
    if backend_name.startswith("mjx"):
        rng = np.random.RandomState(seed)
        compile_s = _run_steps(env, 1, rng, act_dim, seed + 100_000)
        if warmup_steps > 0:
            _run_steps(env, warmup_steps, rng, act_dim, seed + 110_000)

    horizon_times: Dict[int, List[float]] = {h: [] for h in horizons}
    for repeat in range(repeats):
        rng = np.random.RandomState(seed + repeat + 1)
        for horizon in horizons:
            env.reset(seed=seed + repeat + horizon)
            dt = _run_steps(env, horizon, rng, act_dim, seed + repeat + horizon * 10)
            horizon_times[horizon].append(dt)

    env.close()

    horizon_summary = {}
    for horizon, dts in horizon_times.items():
        mean_s = statistics.fmean(dts)
        stdev_s = statistics.pstdev(dts) if len(dts) > 1 else 0.0
        horizon_summary[horizon] = {
            "mean_s": mean_s,
            "stdev_s": stdev_s,
            "steps_per_s": horizon / mean_s if mean_s > 0 else 0.0,
            "effective_steps": float(horizon),
        }

    return {
        "backend": backend_name,
        "compile_s": compile_s,
        "horizons": horizon_summary,
        "effective_multiplier": 1.0,
    }


def _run_mjx_scan_once(
    env,
    steps: int,
    act_dim: int,
    seed: int,
) -> float:
    rng = np.random.RandomState(seed)
    a0 = env._jnp.asarray(rng.uniform(-1.0, 1.0, size=(steps, act_dim)).astype(np.float32))
    a1 = env._jnp.asarray(rng.uniform(-1.0, 1.0, size=(steps, act_dim)).astype(np.float32))
    state0 = env.get_jax_state()

    t0 = time.perf_counter()
    state1 = env.rollout_jax(a0, a1, state=state0, commit=False)
    env._jax.block_until_ready(state1.data.qpos)
    return time.perf_counter() - t0


def _run_mjx_scan_batched_once(
    env,
    steps: int,
    act_dim: int,
    batch_size: int,
    seed: int,
) -> float:
    rng = np.random.RandomState(seed)
    a0 = env._jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(batch_size, steps, act_dim)).astype(np.float32)
    )
    a1 = env._jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(batch_size, steps, act_dim)).astype(np.float32)
    )
    state_b = env.make_batched_state(batch_size=batch_size)

    t0 = time.perf_counter()
    state1_b = env.rollout_jax_batched(a0, a1, state_batched=state_b)
    env._jax.block_until_ready(state1_b.data.qpos)
    return time.perf_counter() - t0


def _benchmark_mjx_scan(
    task: str,
    frame_skip: int,
    physics_profile: str,
    horizons: List[int],
    repeats: int,
    warmup_steps: int,
    seed: int,
    batch_size: int,
    batched: bool,
) -> Dict[str, object]:
    if MJXHumanoidCollabEnv is None:
        raise RuntimeError("MJX backend is unavailable. Install with: pip install -e '.[mjx]'")

    env = MJXHumanoidCollabEnv(
        task=task,
        horizon=max(horizons) + warmup_steps + 1000,
        frame_skip=frame_skip,
        physics_profile=physics_profile,
        detailed_info=False,
    )
    env.reset(seed=seed)
    act_dim = int(env.action_space("h0").shape[0])

    if batched and batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # compile+first step
    if batched:
        compile_s = _run_mjx_scan_batched_once(env, steps=1, act_dim=act_dim, batch_size=batch_size, seed=seed + 100_000)
    else:
        compile_s = _run_mjx_scan_once(env, steps=1, act_dim=act_dim, seed=seed + 100_000)

    # warmup after compile
    if warmup_steps > 0:
        if batched:
            _run_mjx_scan_batched_once(
                env,
                steps=warmup_steps,
                act_dim=act_dim,
                batch_size=batch_size,
                seed=seed + 110_000,
            )
        else:
            _run_mjx_scan_once(
                env,
                steps=warmup_steps,
                act_dim=act_dim,
                seed=seed + 110_000,
            )

    horizon_times: Dict[int, List[float]] = {h: [] for h in horizons}
    for repeat in range(repeats):
        for horizon in horizons:
            env.reset(seed=seed + repeat + horizon)
            if batched:
                dt = _run_mjx_scan_batched_once(
                    env,
                    steps=horizon,
                    act_dim=act_dim,
                    batch_size=batch_size,
                    seed=seed + repeat + horizon * 10,
                )
            else:
                dt = _run_mjx_scan_once(
                    env,
                    steps=horizon,
                    act_dim=act_dim,
                    seed=seed + repeat + horizon * 10,
                )
            horizon_times[horizon].append(dt)

    env.close()

    multiplier = float(batch_size if batched else 1)
    horizon_summary = {}
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

    backend_name = "mjx-scan-batched" if batched else "mjx-scan"
    return {
        "backend": backend_name,
        "compile_s": compile_s,
        "horizons": horizon_summary,
        "effective_multiplier": multiplier,
        "batch_size": batch_size if batched else 1,
    }


def _estimate_break_even(
    cpu_results: Dict[str, object],
    mjx_results: Dict[str, object],
) -> Optional[float]:
    """Estimate steps needed for MJX to amortize compile overhead."""
    compile_s = mjx_results.get("compile_s")
    if compile_s is None:
        return None

    # Break-even only meaningful for single-env equivalent throughput.
    if float(mjx_results.get("effective_multiplier", 1.0)) != 1.0:
        return None

    cpu_horizons: Dict[int, Dict[str, float]] = cpu_results["horizons"]  # type: ignore[assignment]
    mjx_horizons: Dict[int, Dict[str, float]] = mjx_results["horizons"]  # type: ignore[assignment]
    common_h = sorted(set(cpu_horizons.keys()) & set(mjx_horizons.keys()))
    if not common_h:
        return None

    h = common_h[-1]
    cpu_step_s = cpu_horizons[h]["mean_s"] / cpu_horizons[h]["effective_steps"]
    mjx_step_s = mjx_horizons[h]["mean_s"] / mjx_horizons[h]["effective_steps"]
    delta = cpu_step_s - mjx_step_s
    if delta <= 0:
        return None

    return float(compile_s / delta)


def _print_results(results: Dict[str, object]) -> None:
    backend = str(results["backend"])
    compile_s = results.get("compile_s")
    horizons: Dict[int, Dict[str, float]] = results["horizons"]  # type: ignore[assignment]
    batch_size = int(results.get("batch_size", 1))

    print(f"\n[{backend.upper()}]")
    if compile_s is not None:
        print(f"compile+first_step: {compile_s:.4f}s")
    if batch_size > 1:
        print(f"batch_size: {batch_size}")
    for h in sorted(horizons.keys()):
        row = horizons[h]
        print(
            f"steps={h:>6d} | effective_steps={int(row['effective_steps']):>8d} "
            f"| mean={row['mean_s']:.4f}s | std={row['stdev_s']:.4f}s "
            f"| throughput={row['steps_per_s']:.1f} steps/s"
        )


def parse_args() -> argparse.Namespace:
    physics_profiles = available_physics_profiles()

    parser = argparse.ArgumentParser(description="Benchmark HumanoidCollab CPU vs MJX backend.")
    parser.add_argument("--task", type=str, default="hug", choices=["hug", "handshake", "box_lift"])
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["cpu", "mjx", "both"],
        help="Which backend(s) to benchmark.",
    )
    parser.add_argument(
        "--mjx-mode",
        type=str,
        default="pettingzoo",
        choices=["pettingzoo", "scan", "scan-batched"],
        help="MJX benchmark mode. Use scan-batched for max throughput.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for --mjx-mode scan-batched.",
    )
    parser.add_argument(
        "--batch-sweep",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of batch sizes for scan-batched sweep (e.g. 64 128 256 512).",
    )
    parser.add_argument(
        "--cpu-physics-profile",
        type=str,
        default="default",
        choices=physics_profiles,
        help="Physics profile for CPU backend.",
    )
    parser.add_argument(
        "--mjx-physics-profile",
        type=str,
        default="train_fast",
        choices=physics_profiles,
        help="Physics profile for MJX backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))

    run_cpu = args.backend in {"cpu", "both"}
    run_mjx = args.backend in {"mjx", "both"}

    cpu_results = None
    mjx_results = None

    if run_cpu:
        cpu_results = _benchmark_backend_step_loop(
            env_cls=HumanoidCollabEnv,
            backend_name="cpu",
            task=args.task,
            frame_skip=args.frame_skip,
            physics_profile=args.cpu_physics_profile,
            horizons=horizons,
            repeats=args.repeats,
            warmup_steps=0,
            seed=args.seed,
        )
        _print_results(cpu_results)

    if run_mjx:
        if MJXHumanoidCollabEnv is None:
            raise RuntimeError(
                "MJX backend is unavailable. Install with: pip install -e '.[mjx]'"
            )
        try:
            if args.mjx_mode == "pettingzoo":
                mjx_results = _benchmark_backend_step_loop(
                    env_cls=MJXHumanoidCollabEnv,
                    backend_name="mjx-pettingzoo",
                    task=args.task,
                    frame_skip=args.frame_skip,
                    physics_profile=args.mjx_physics_profile,
                    horizons=horizons,
                    repeats=args.repeats,
                    warmup_steps=args.warmup_steps,
                    seed=args.seed,
                )
            elif args.mjx_mode == "scan":
                mjx_results = _benchmark_mjx_scan(
                    task=args.task,
                    frame_skip=args.frame_skip,
                    physics_profile=args.mjx_physics_profile,
                    horizons=horizons,
                    repeats=args.repeats,
                    warmup_steps=args.warmup_steps,
                    seed=args.seed,
                    batch_size=1,
                    batched=False,
                )
            else:
                if args.batch_sweep:
                    sweep_sizes = sorted({int(v) for v in args.batch_sweep if int(v) > 0})
                    if not sweep_sizes:
                        raise ValueError("batch_sweep must include at least one positive integer.")
                    sweep_results = []
                    for bs in sweep_sizes:
                        result = _benchmark_mjx_scan(
                            task=args.task,
                            frame_skip=args.frame_skip,
                            physics_profile=args.mjx_physics_profile,
                            horizons=horizons,
                            repeats=args.repeats,
                            warmup_steps=args.warmup_steps,
                            seed=args.seed,
                            batch_size=bs,
                            batched=True,
                        )
                        _print_results(result)
                        sweep_results.append(result)
                    target_h = horizons[-1]
                    best = max(
                        sweep_results,
                        key=lambda r: r["horizons"][target_h]["steps_per_s"],  # type: ignore[index]
                    )
                    best_bs = int(best.get("batch_size", -1))
                    best_tput = float(best["horizons"][target_h]["steps_per_s"])  # type: ignore[index]
                    print(
                        f"\nBest scan-batched config at horizon={target_h}: "
                        f"batch_size={best_bs}, throughput={best_tput:.1f} steps/s"
                    )
                    mjx_results = best
                else:
                    mjx_results = _benchmark_mjx_scan(
                        task=args.task,
                        frame_skip=args.frame_skip,
                        physics_profile=args.mjx_physics_profile,
                        horizons=horizons,
                        repeats=args.repeats,
                        warmup_steps=args.warmup_steps,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        batched=True,
                    )
        except ImportError as exc:
            raise RuntimeError(
                f"MJX backend is unavailable: {exc}. Install with: pip install -e '.[mjx]'"
            ) from exc
        if not (args.mjx_mode == "scan-batched" and args.batch_sweep):
            _print_results(mjx_results)

    if cpu_results is not None and mjx_results is not None:
        break_even = _estimate_break_even(cpu_results, mjx_results)
        if break_even is None:
            print("\nBreak-even estimate: not reached or not comparable for this mode.")
        else:
            print(f"\nEstimated break-even: ~{break_even:.0f} steps")


if __name__ == "__main__":
    main()
