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
for i in range(100):
    ac = np.zeros(10)
    obs_vf, obs_pol, rew, done, info = env.step(ac)
    pos_err = obs_pol[0][-2:]
    ob_cmd = obs_pol[0][-8:-2] # ob_command is length 6
    if i % 10 == 0:
        print(f"Step {i}:")
        print(f"  Target Pos Global: {env.ref_dict['base_pos_global'][:2]}")
        print(f"  Robot Pos Global: {env.qpos[:2]}")
        print(f"  pos_err_local: {pos_err}")
        print(f"  ob_command: {ob_cmd}")
        print(f"  local_vel target: {env.ref_dict['base_vel_local']}")
