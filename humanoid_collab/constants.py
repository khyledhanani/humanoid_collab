"""Project-wide constants."""

# Default root joint Z height (meters) for a standing humanoid with feet on the floor (z=0).
# This value is chosen so that the foot geoms touch the ground at reset instead of starting
# in the air and "falling" during the first few physics steps.
DEFAULT_ROOT_Z = 1.02

