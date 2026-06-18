import sys
import numpy as np
sys.path.append('.')
from mujoco_env.reference_generator import ReferenceGenerator
config = {
    "use_footstep_plan": True,
    "add_standing": False,
    "fixed_gait": False,
    "add_rotation": False
}
rg = ReferenceGenerator(300, 0.03333333, config)
for i in range(10):
    rg.update_ref_env(i*0.033, [0,0], 0.0)
    print("global:", rg.footstep_traj[rg.footstep_index-1], "local vel:", rg.last_ref_gaitparams)
