import sys
import numpy as np
sys.path.append('.')
from mujoco_env.cassie_env import CassieEnv
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
obs_vf, obs_pol = env.reset()
for i in range(10):
    ac = np.zeros(10)
    obs_vf, obs_pol, rew, done, info = env.step(ac)
    pos_err = obs_pol[0][-2:]
    curr_xy = env.qpos[[0, 1]] - env.init_xy
    pos_err_global = env.ref_dict["base_pos_global"][[0, 1]] - curr_xy
    print(f"Step {i}:")
    print(f"  Target Pos Global: {env.ref_dict['base_pos_global'][:2]}")
    print(f"  Robot qpos: {env.qpos[:2]}")
    print(f"  Robot init_xy: {env.init_xy}")
    print(f"  curr_xy: {curr_xy}")
    print(f"  pos_err_global: {pos_err_global}")
    print(f"  pos_err_local (from env): {pos_err}")
    print(f"  Robot Yaw: {env.curr_rpy_obs[-1]}")
