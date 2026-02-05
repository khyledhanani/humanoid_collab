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

# With MJX backend dependencies:
pip install -e ".[mjx]"
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
from humanoid_collab import MJXHumanoidCollabEnv

env = HumanoidCollabEnv(task="hug", stage=0, horizon=1000)
obs, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()

# MJX backend (physics stepped with MJX)
mjx_env = MJXHumanoidCollabEnv(task="hug", stage=0, horizon=1000)
obs, infos = mjx_env.reset(seed=42)
while mjx_env.agents:
    actions = {agent: mjx_env.action_space(agent).sample() for agent in mjx_env.agents}
    obs, rewards, terminations, truncations, infos = mjx_env.step(actions)
mjx_env.close()
```

### Backend selection in scripts

```bash
python scripts/rollout_random.py --task hug --backend cpu
python scripts/rollout_random.py --task hug --backend mjx
python scripts/render_demo.py --task handshake --backend mjx --mode video
python scripts/benchmark_backends.py --task hug --backend both --horizons 100 1000 10000 --repeats 3
```

`MJXHumanoidCollabEnv` runs step logic fully on-device (MJX physics + JAX obs/reward/success/contact proxy). CPU sync is only used for rendering and `state()`.

## Environment API

The environment implements PettingZoo's `ParallelEnv` with two agents (`h0`, `h1`).

- **Observation space**: `Box(low=-inf, high=inf, shape=(obs_dim,), dtype=float32)`
  - Base obs (~41 dims): proprioception + partner relative features
  - Task-specific obs: varies by task (3 for hug, 5 for handshake, 8 for box_lift)
- **Action space**: `Box(low=-1, high=1, shape=(18,), dtype=float32)` — 18 actuators per humanoid
- **Reward**: Cooperative (same scalar for both agents)

### Constructor

```python
HumanoidCollabEnv(
    task="hug",          # Task name: "hug", "handshake", "box_lift"
    render_mode=None,    # "human" or "rgb_array"
    horizon=1000,        # Max steps per episode
    frame_skip=5,        # Physics steps per action
    hold_target=30,      # Success hold duration
    stage=0,             # Curriculum stage
)
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
  mjx_env.py                # MJXHumanoidCollabEnv (on-device JAX step path)
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
