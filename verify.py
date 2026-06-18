import sys
import numpy as np
import time
import os
sys.path.append(os.path.abspath('.'))

from configs.env_config import config_train
config = config_train.copy()
config["is_visual"] = False
config["use_footstep_plan"] = True

from mujoco_env.cassie_env import CassieEnv

def main():
    env = CassieEnv(config)
    obs = env.reset()
    
    print("Reset done. Running 20 steps...")
    for i in range(20):
        obs_vf, obs_pol, reward, done, info = env.step(np.zeros(10))
        
        ref_pos = env.ref_dict["base_pos_global"]
        curr_xy = env.qpos[[0, 1]] - env.init_xy
        pos_err_global = ref_pos[[0, 1]] - curr_xy
        curr_yaw = env.curr_rpy_obs[-1]
        
        local_v = env.ref_dict["base_vel_local"]
        
        print(f"Step {i+1}: Target POS: {ref_pos[[0,1]]}, Pos Err G: {pos_err_global}, Local Vel: {local_v}, Target Yaw: {env.ref_dict['base_rot_global'][-1]}")
        
        if done:
            print("Episode done early!")
            break

if __name__ == "__main__":
    main()
