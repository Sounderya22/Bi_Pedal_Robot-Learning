import numpy as np
from mujoco_env.gait_library import GaitLibrary

gl = GaitLibrary(secs_per_env_step=0.033)
p, ct = gl._get_ref_gait([0.372, 0.344, 0.9], stanceLeg=1)
print(f"For vx=0.372, vy=0.344, ct = {ct}")
