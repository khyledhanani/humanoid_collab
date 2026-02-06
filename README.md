# Humanoid Collab

A generalized two-agent humanoid collaboration environment using MuJoCo and PettingZoo. Supports multiple cooperative tasks with a unified API.

## Tasks

| Task | Description | Success Condition |
|------|-------------|-------------------|
| **hug** | Two agents approach, embrace, and hold a stable hug | Mutual arm-torso contact + facing + proximity + stability for K steps |
| **handshake** | Two agents approach, extend hands, and shake | Hand-to-hand contact + facing + arm's-length distance + stability for K steps |
| **box_lift** | Two agents cooperatively grip and lift a box | Box at target height + both gripping + stable + upright for K steps |

## Installation

```bash
cd humanoid_collab
pip install -e .

# With dev dependencies (for tests):
pip install -e ".[dev]"

# With training dependencies:
pip install -e ".[train]"
```

## Dependencies

- `mujoco>=3.0.0`
- `numpy>=1.21.0`
- `pettingzoo>=1.24.0`
- `gymnasium>=0.29.0`

## Quick Start

### Run random rollout

```bash
python scripts/rollout_random.py --task hug
python scripts/rollout_random.py --task handshake
python scripts/rollout_random.py --task box_lift --stage 2
```

### Render demo

```bash
# Save video
python scripts/render_demo.py --task box_lift --mode video --output box_demo.mp4

# Interactive viewer (requires mjpython on macOS)
python scripts/render_demo.py --task hug --mode human
```

### Use in code

```python
from humanoid_collab import HumanoidCollabEnv

env = HumanoidCollabEnv(task="hug", stage=0, horizon=1000)
obs, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

### Script examples

```bash
python scripts/rollout_random.py --task handshake --fixed-standing --control-mode arms_only
python scripts/render_demo.py --task handshake --mode video --fixed-standing --control-mode arms_only
python scripts/benchmark_backends.py --task hug --mode both --horizons 100 1000 10000 --repeats 3
python scripts/benchmark_backends.py --task hug --mode subproc --num-envs 8 --horizons 10000 --repeats 3
python scripts/train_ippo.py --task handshake --backend cpu --fixed-standing --control-mode arms_only --physics-profile default --total-steps 500000 --rollout-steps 1024 --ppo-epochs 8 --minibatch-size 256
python scripts/render_ippo.py --checkpoint checkpoints/ippo_handshake_fixed_arms/ippo_update_000100.pt --episodes 3 --deterministic
python scripts/train_maddpg.py --task handshake --backend cpu --fixed-standing --control-mode arms_only --stage 1 --num-envs 8 --total-steps 800000 --device cuda
python scripts/render_maddpg.py --checkpoint checkpoints/maddpg_handshake_fixed_arms/maddpg_step_0800000.pt --episodes 3
```

Available physics profiles: `default`, `balanced`, `train_fast`.
Control modes: `all`, `arms_only`.

CPU parallelization:
- `SubprocHumanoidCollabVecEnv` in `humanoid_collab/vector_env.py` runs many CPU env instances in subprocesses.
- Use `--mode subproc --num-envs N` in `scripts/benchmark_backends.py` to measure multi-env CPU throughput.

IPPO training:
- `scripts/train_ippo.py` trains two independent PPO policies (`h0`, `h1`) on this environment.
- For fixed-standing handshake arm training, use `--task handshake --fixed-standing --control-mode arms_only`.
- Auto-curriculum is available via `--auto-curriculum`.

MADDPG training:
- `scripts/train_maddpg.py` trains two deterministic policies with centralized critics (CTDE).
- Actors only consume each agent's local observation; critics consume joint observations/actions during training.
- Use GPU for model updates via `--device cuda` (env simulation remains CPU MuJoCo).

## Environment API

The environment implements PettingZoo's `ParallelEnv` with two agents (`h0`, `h1`).

- **Observation space**:
  - `observation_mode="proprio"`: `Box(low=-inf, high=inf, shape=(47,), dtype=float32)`
    - Self-only egocentric proprioception (no partner/object state in vector obs)
    - Task-specific vector obs currently disabled (`task_obs_dim = 0` for all tasks)
  - `observation_mode="rgb"`: `Box(low=0, high=255, shape=(H, W, 3), dtype=uint8)`
    - Per-agent egocentric front camera (`h0_ego`, `h1_ego`)
  - `observation_mode="gray"`: `Box(low=0, high=255, shape=(H, W, 1), dtype=uint8)`
    - Grayscale egocentric camera observations
- **Action space**: `Box(low=-1, high=1, shape=(18,), dtype=float32)` by default
  - In `control_mode="arms_only"`, action dim is 6 per agent
- **Reward**: Cooperative (same scalar for both agents)

### Constructor

```python
HumanoidCollabEnv(
    task="hug",          # Task name: "hug", "handshake", "box_lift"
    render_mode=None,     # "human" or "rgb_array"
    horizon=1000,         # Max steps per episode
    frame_skip=5,         # Physics steps per action
    hold_target=30,       # Success hold duration
    stage=0,              # Curriculum stage
    physics_profile="default",
    fixed_standing=False,
    control_mode="all",
    observation_mode="proprio",  # "proprio", "rgb", or "gray"
    obs_rgb_width=84,            # used when observation_mode is rgb/gray
    obs_rgb_height=84,           # used when observation_mode is rgb/gray
)
```

Example RGB mode:

```python
env = HumanoidCollabEnv(task="handshake", observation_mode="rgb", obs_rgb_width=96, obs_rgb_height=96)
obs, infos = env.reset(seed=42)
print(obs["h0"].shape)  # (96, 96, 3)
```

Example grayscale mode:

```python
env = HumanoidCollabEnv(task="handshake", observation_mode="gray", obs_rgb_width=96, obs_rgb_height=96)
obs, infos = env.reset(seed=42)
print(obs["h0"].shape)  # (96, 96, 1)
```

### Reset options

```python
obs, infos = env.reset(seed=42, options={"stage": 2})
```

## Reward Structure

All tasks share common reward patterns:

### Shaping terms (guide learning)
- **Distance/approach**: Exponential decay toward target proximity
- **Facing alignment**: `dot(fwd_self, -fwd_partner)` for face-to-face tasks
- **Stability**: Penalize high relative velocity

### Task-specific terms
- **Contact rewards**: Binary bonuses for task-relevant contacts
- **Proximity shaping**: Guide hands toward targets (partner's back, partner's hand, box)
- **Lift reward** (box_lift only): Progress toward target height

### Penalties (anti-reward-hacking)
- **Energy**: `-c * sum(ctrl^2)`
- **Impact**: `-c * contact_force_proxy`
- **Fall**: Large negative on fall termination

### Terminal bonus
- Large positive reward when success condition held for `hold_target` steps

## Curriculum Stages

Each task defines progressive curriculum stages that adjust reward weights:

### Hug (4 stages)
| Stage | Focus |
|-------|-------|
| 0 | Approach + facing only |
| 1 | Add hand-to-back shaping |
| 2 | Add arm-torso contact reward |
| 3 | Full hug with higher success bonus |

### Handshake (3 stages)
| Stage | Focus |
|-------|-------|
| 0 | Approach + facing only |
| 1 | Hand proximity shaping |
| 2 | Full handshake with contact reward |

### Box Lift (4 stages)
| Stage | Focus |
|-------|-------|
| 0 | Approach box |
| 1 | Hand-to-box shaping |
| 2 | Grip contact + initial lift reward |
| 3 | Full lift to target height + hold |

Switch stages at reset: `env.reset(options={"stage": 2})`

## Architecture

```
humanoid_collab/
  env.py                    # HumanoidCollabEnv (PettingZoo ParallelEnv)
  vector_env.py             # SubprocHumanoidCollabVecEnv (CPU subprocess vectorization)
  mjcf_builder.py           # Programmatic MJCF XML generation
  tasks/
    base.py                 # Abstract TaskConfig interface
    registry.py             # Task registry
    hug.py                  # Hug task implementation
    handshake.py            # Handshake task implementation
    box_lift.py             # Box lift task implementation
  utils/
    ids.py                  # ID caching with dynamic geom groups
    kinematics.py           # Coordinate transforms
    contacts.py             # Generalized contact detection
    obs.py                  # Base observation builder
```

### Adding a new task

1. Create `tasks/my_task.py` implementing `TaskConfig`
2. Use `@register_task` decorator
3. Import in `tasks/registry.py`
4. The task is automatically available via `HumanoidCollabEnv(task="my_task")`

## Tests

```bash
pytest tests/ -v
```
