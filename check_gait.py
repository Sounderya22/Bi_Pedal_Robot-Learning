import numpy as np
from mujoco_env.gait_library import GaitLibrary

gl = GaitLibrary(secs_per_env_step=0.033)
print(f"gait_params_min: {gl.gait_params_min}")
print(f"gait_params_max: {gl.gait_params_max}")
print(f"Velocity grid shapes: {gl.library['Velocity'].shape}")
# Let's check what joint positions are for vx=0.5 vs vx=-0.5
p1 = gl.get_ref_states([0.5, 0.0, 0.9])
p2 = gl.get_ref_states([-0.5, 0.0, 0.9])
print(f"Norm of diff between vx=0.5 and vx=-0.5: {np.linalg.norm(p1 - p2):.3f}")

