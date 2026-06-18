import sys
sys.path.append('.')
from mujoco_env.cassie_env import CassieEnv
import numpy as np
import math

def patched_get_observation(self, acs=np.zeros(10), step=False):
    obs_vf, obs_pol = self.original_get_observation(acs, step)
    print(f"INSIDE get_observation: qpos={self.qpos[[0,1]]}, init_xy={self.init_xy}")
    curr_xy = self.qpos[[0, 1]] - self.init_xy
    pos_err_global = self.ref_dict["base_pos_global"][[0, 1]] - curr_xy
    curr_yaw = self.curr_rpy_obs[-1]
    pos_err_local_x = pos_err_global[0] * math.cos(curr_yaw) + pos_err_global[1] * math.sin(curr_yaw)
    pos_err_local_y = -pos_err_global[0] * math.sin(curr_yaw) + pos_err_global[1] * math.cos(curr_yaw)
    print(f"INSIDE get_observation: curr_xy={curr_xy}, pos_err_global={pos_err_global}, pos_err_local=[{pos_err_local_x}, {pos_err_local_y}]")
    print(f"obs_pol[-2:]={obs_pol[0][-2:]}")
    return obs_vf, obs_pol

CassieEnv.original_get_observation = CassieEnv._CassieEnv__get_observation
CassieEnv._CassieEnv__get_observation = patched_get_observation

config = {
    "max_timesteps": 300,
    "is_visual": False,
    "minimal_rand": True,
    "is_noisy": False,
    "add_standing": False,
    "use_footstep_plan": True,
    "fixed_gait": False,
    "add_rotation": False
}
env = CassieEnv(config)
env.reset()
env.step(np.zeros(10))
