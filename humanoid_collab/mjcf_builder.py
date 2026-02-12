"""Programmatic MJCF XML generation for the humanoid collaboration environment."""

from typing import Dict, Optional


_PHYSICS_PROFILES: Dict[str, Dict[str, str]] = {
    # Highest-fidelity baseline used by the original environment.
    "default": {
        "timestep": "0.005",
        "iterations": "50",
        "tolerance": "1e-10",
        "solver": "Newton",
        "jacobian": "dense",
        "cone": "pyramidal",
    },
    # Moderate speed/fidelity trade-off.
    "balanced": {
        "timestep": "0.005",
        "iterations": "20",
        "tolerance": "1e-7",
        "solver": "CG",
        "jacobian": "sparse",
        "cone": "pyramidal",
    },
    # Aggressive speed profile for training throughput.
    "train_fast": {
        "timestep": "0.005",
        "iterations": "12",
        "tolerance": "1e-6",
        "solver": "CG",
        "jacobian": "sparse",
        "cone": "pyramidal",
    },
}


def available_physics_profiles():
    """Return supported physics profile names."""
    return sorted(_PHYSICS_PROFILES.keys())


def _humanoid_body(
    prefix: str,
    pos: str,
    material: str,
    quat: Optional[str] = None,
) -> str:
    """Generate the XML for one humanoid body.

    Args:
        prefix: Agent prefix (e.g., 'h0' or 'h1')
        pos: Starting position as 'x y z' string
        material: Material name for coloring

    Returns:
        XML string for the humanoid body
    """
    quat_attr = f' quat="{quat}"' if quat is not None else ""
    return f"""
    <body name="{prefix}_torso" pos="{pos}"{quat_attr}>
      <camera name="{prefix}_track" mode="trackcom" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
      <camera name="{prefix}_ego" mode="fixed" pos="0 0 0.22" xyaxes="0 -1 0 0 0 1" fovy="80"/>
      <freejoint name="{prefix}_root"/>

      <geom name="{prefix}_torso_geom" fromto="0 -0.07 0 0 0.07 0" size="0.07" type="capsule" material="{material}"/>
      <geom name="{prefix}_chest_geom" fromto="0 -0.06 0.15 0 0.06 0.15" size="0.06" type="capsule" material="{material}"/>
      <geom name="{prefix}_abdomen_geom" fromto="0 -0.05 -0.1 0 0.05 -0.1" size="0.05" type="capsule" material="{material}"/>

      <site name="{prefix}_chest" pos="0 0 0.15" size="0.02" rgba="1 0 0 0.5"/>
      <site name="{prefix}_pelvis" pos="0 0 -0.2" size="0.02" rgba="1 0 0 0.5"/>
      <site name="{prefix}_back" pos="-0.08 0 0.05" size="0.02" rgba="1 0 0 0.5"/>

      <body name="{prefix}_head" pos="0 0 0.25">
        <joint name="{prefix}_head_yaw" type="hinge" axis="0 0 1" range="-50 50" armature="0.02" damping="0.1" stiffness="5"/>
        <joint name="{prefix}_head_pitch" type="hinge" axis="0 1 0" range="-30 30" armature="0.02" damping="0.1" stiffness="5"/>
        <geom name="{prefix}_head_geom" type="sphere" size="0.09" material="{material}"/>
        <site name="{prefix}_head_site" pos="0 0 0.05" size="0.02" rgba="1 0 0 0.5"/>
      </body>

      <body name="{prefix}_left_upper_arm" pos="0 0.17 0.12">
        <joint name="{prefix}_left_shoulder1" type="hinge" axis="2 1 1" range="-85 60" armature="0.0068"/>
        <joint name="{prefix}_left_shoulder2" type="hinge" axis="0 -1 1" range="-85 60" armature="0.0051"/>
        <geom name="{prefix}_left_upper_arm_geom" fromto="0 0 0 0.16 0.16 -0.16" size="0.04" type="capsule" material="{material}"/>

        <body name="{prefix}_left_lower_arm" pos="0.16 0.16 -0.16">
          <joint name="{prefix}_left_elbow" type="hinge" axis="0 -1 1" range="-90 50" stiffness="0" armature="0.0028"/>
          <geom name="{prefix}_left_lower_arm_geom" fromto="0.01 0.01 0.01 0.17 0.17 0.17" size="0.031" type="capsule" material="{material}"/>

          <body name="{prefix}_left_hand" pos="0.18 0.18 0.18">
            <geom name="{prefix}_left_hand_geom" type="sphere" size="0.04" material="{material}"/>
            <site name="{prefix}_lhand" pos="0 0 0" size="0.02" rgba="0 1 0 0.5"/>
          </body>
        </body>
      </body>

      <body name="{prefix}_right_upper_arm" pos="0 -0.17 0.12">
        <joint name="{prefix}_right_shoulder1" type="hinge" axis="2 -1 1" range="-60 85" armature="0.0068"/>
        <joint name="{prefix}_right_shoulder2" type="hinge" axis="0 1 1" range="-60 85" armature="0.0051"/>
        <geom name="{prefix}_right_upper_arm_geom" fromto="0 0 0 0.16 -0.16 -0.16" size="0.04" type="capsule" material="{material}"/>

        <body name="{prefix}_right_lower_arm" pos="0.16 -0.16 -0.16">
          <joint name="{prefix}_right_elbow" type="hinge" axis="0 -1 -1" range="-90 50" stiffness="0" armature="0.0028"/>
          <geom name="{prefix}_right_lower_arm_geom" fromto="0.01 -0.01 0.01 0.17 -0.17 0.17" size="0.031" type="capsule" material="{material}"/>

          <body name="{prefix}_right_hand" pos="0.18 -0.18 0.18">
            <geom name="{prefix}_right_hand_geom" type="sphere" size="0.04" material="{material}"/>
            <site name="{prefix}_rhand" pos="0 0 0" size="0.02" rgba="0 1 0 0.5"/>
          </body>
        </body>
      </body>

      <body name="{prefix}_lower_body" pos="0 0 -0.2">
        <joint name="{prefix}_abdomen_z" type="hinge" axis="0 0 1" range="-45 45" armature="0.02"/>
        <joint name="{prefix}_abdomen_y" type="hinge" axis="0 1 0" range="-75 30" armature="0.02"/>
        <geom name="{prefix}_pelvis_geom" fromto="-0.02 -0.07 0 -0.02 0.07 0" size="0.09" type="capsule" material="{material}"/>

        <body name="{prefix}_left_thigh" pos="0 0.1 -0.04">
          <joint name="{prefix}_left_hip_x" type="hinge" axis="1 0 0" range="-25 5" armature="0.01"/>
          <joint name="{prefix}_left_hip_z" type="hinge" axis="0 0 1" range="-60 35" armature="0.01"/>
          <joint name="{prefix}_left_hip_y" type="hinge" axis="0 1 0" range="-110 20" armature="0.01"/>
          <geom name="{prefix}_left_thigh_geom" fromto="0 0 0 0 0.01 -0.34" size="0.06" type="capsule" material="{material}"/>

          <body name="{prefix}_left_shin" pos="0 0.01 -0.403">
            <joint name="{prefix}_left_knee" type="hinge" axis="0 1 0" range="-160 -2" armature="0.006"/>
            <geom name="{prefix}_left_shin_geom" fromto="0 0 0 0 0 -0.3" size="0.049" type="capsule" material="{material}"/>

            <body name="{prefix}_left_foot" pos="0 0 -0.35">
              <joint name="{prefix}_left_ankle" type="hinge" axis="0 1 0" range="-50 50" armature="0.008"/>
              <geom name="{prefix}_left_foot_geom" fromto="-0.07 -0.02 0 0.14 -0.04 0" size="0.027" type="capsule" material="{material}"/>
            </body>
          </body>
        </body>

        <body name="{prefix}_right_thigh" pos="0 -0.1 -0.04">
          <joint name="{prefix}_right_hip_x" type="hinge" axis="-1 0 0" range="-25 5" armature="0.01"/>
          <joint name="{prefix}_right_hip_z" type="hinge" axis="0 0 -1" range="-60 35" armature="0.01"/>
          <joint name="{prefix}_right_hip_y" type="hinge" axis="0 1 0" range="-110 20" armature="0.01"/>
          <geom name="{prefix}_right_thigh_geom" fromto="0 0 0 0 -0.01 -0.34" size="0.06" type="capsule" material="{material}"/>

          <body name="{prefix}_right_shin" pos="0 -0.01 -0.403">
            <joint name="{prefix}_right_knee" type="hinge" axis="0 1 0" range="-160 -2" armature="0.006"/>
            <geom name="{prefix}_right_shin_geom" fromto="0 0 0 0 0 -0.3" size="0.049" type="capsule" material="{material}"/>

            <body name="{prefix}_right_foot" pos="0 0 -0.35">
              <joint name="{prefix}_right_ankle" type="hinge" axis="0 1 0" range="-50 50" armature="0.008"/>
              <geom name="{prefix}_right_foot_geom" fromto="-0.07 0.02 0 0.14 0.04 0" size="0.027" type="capsule" material="{material}"/>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _humanoid_actuators(prefix: str) -> str:
    """Generate actuator XML for one humanoid.

    Args:
        prefix: Agent prefix (e.g., 'h0' or 'h1')

    Returns:
        XML string for actuators
    """
    return f"""
    <motor name="{prefix}_abdomen_z" joint="{prefix}_abdomen_z" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_abdomen_y" joint="{prefix}_abdomen_y" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_hip_x" joint="{prefix}_left_hip_x" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_hip_z" joint="{prefix}_left_hip_z" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_hip_y" joint="{prefix}_left_hip_y" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_knee" joint="{prefix}_left_knee" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_ankle" joint="{prefix}_left_ankle" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_hip_x" joint="{prefix}_right_hip_x" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_hip_z" joint="{prefix}_right_hip_z" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_hip_y" joint="{prefix}_right_hip_y" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_knee" joint="{prefix}_right_knee" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_ankle" joint="{prefix}_right_ankle" gear="100" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_shoulder1" joint="{prefix}_left_shoulder1" gear="25" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_shoulder2" joint="{prefix}_left_shoulder2" gear="25" ctrlrange="-1 1"/>
    <motor name="{prefix}_left_elbow" joint="{prefix}_left_elbow" gear="25" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_shoulder1" joint="{prefix}_right_shoulder1" gear="25" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_shoulder2" joint="{prefix}_right_shoulder2" gear="25" ctrlrange="-1 1"/>
    <motor name="{prefix}_right_elbow" joint="{prefix}_right_elbow" gear="25" ctrlrange="-1 1"/>"""


def _fixed_standing_equality(mode: str = "torso") -> str:
    """Constraint block for fixed-standing setups.

    Args:
        mode: "torso" to weld torso roots, "lower_body" to weld pelvis/lower-body
            segments while allowing torso articulation around abdomen joints.
    """
    if mode == "torso":
        return """
  <equality>
    <weld body1="h0_torso" body2="world"/>
    <weld body1="h1_torso" body2="world"/>
  </equality>"""
    if mode == "lower_body":
        return """
  <equality>
    <weld body1="h0_lower_body" body2="world"/>
    <weld body1="h1_lower_body" body2="world"/>
  </equality>"""
    raise ValueError(f"Unknown fixed_standing mode '{mode}'. Expected 'torso' or 'lower_body'.")


def build_mjcf(
    task_worldbody_additions: str = "",
    task_actuator_additions: str = "",
    physics_profile: str = "default",
    fixed_standing: bool = False,
    fixed_standing_mode: str = "torso",
    spawn_half_distance: float = 1.0,
    h1_faces_h0: bool = False,
) -> str:
    """Build the complete MJCF XML with two humanoids and optional task-specific additions.

    Args:
        task_worldbody_additions: XML fragment to inject into <worldbody> (e.g., a box body).
        task_actuator_additions: XML fragment to inject into <actuator> (e.g., extra actuators).
        physics_profile: Physics option profile name.
        fixed_standing: If True, add weld constraints for fixed-standing configuration.
        fixed_standing_mode: Weld mode when fixed_standing=True:
            - "torso": weld each torso root to world
            - "lower_body": weld each lower body to world
        spawn_half_distance: Spawn offset magnitude from origin for each humanoid torso.
        h1_faces_h0: If True, initialize h1 facing opposite direction.

    Returns:
        Complete MJCF XML string.
    """
    if physics_profile not in _PHYSICS_PROFILES:
        valid = ", ".join(available_physics_profiles())
        raise ValueError(
            f"Unknown physics profile '{physics_profile}'. Available: {valid}"
        )

    physics = _PHYSICS_PROFILES[physics_profile]

    h0_pos = f"{-float(spawn_half_distance)} 0 1.4"
    h1_pos = f"{float(spawn_half_distance)} 0 1.4"
    h0_body = _humanoid_body("h0", h0_pos, "h0_mat")
    h1_quat = "0 0 0 1" if h1_faces_h0 else None
    h1_body = _humanoid_body("h1", h1_pos, "h1_mat", quat=h1_quat)
    h0_actuators = _humanoid_actuators("h0")
    h1_actuators = _humanoid_actuators("h1")
    equality_block = _fixed_standing_equality(fixed_standing_mode) if fixed_standing else ""

    xml = f"""<mujoco model="humanoid_collab">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="{physics['timestep']}" iterations="{physics['iterations']}" tolerance="{physics['tolerance']}" solver="{physics['solver']}" jacobian="{physics['jacobian']}" cone="{physics['cone']}"/>

  <size nstack="3000000" nuser_body="1"/>

  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" friction="0.8 0.1 0.1" margin="0.001" material="geom"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="geom" rgba="0.8 0.6 0.4 1"/>
    <material name="h0_mat" rgba="0.2 0.6 0.8 1"/>
    <material name="h1_mat" rgba="0.8 0.4 0.2 1"/>
  </asset>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular="0.1 0.1 0.1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="plane" material="groundplane"/>

    <!-- Humanoid 0 -->
{h0_body}

    <!-- Humanoid 1 -->
{h1_body}

    <!-- Task-specific additions -->
{task_worldbody_additions}
  </worldbody>

{equality_block}

  <actuator>
    <!-- Humanoid 0 actuators -->
{h0_actuators}

    <!-- Humanoid 1 actuators -->
{h1_actuators}

    <!-- Task-specific actuators -->
{task_actuator_additions}
  </actuator>
</mujoco>"""

    return xml
