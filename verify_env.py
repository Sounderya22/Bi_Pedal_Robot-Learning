import sys
import numpy as np

# Mock config to avoid visualizer
config = {
    "max_timesteps": 2500,
    "is_visual": False,
    "minimal_rand": False,
    "is_noisy": False,
    "add_standing": True,
    "fixed_gait": True,
    "add_rotation": True,
    "fixed_gait_cmd": [0.0, 0.0, 0.98, 0.0],
    "swing_duration": 0.25,
    "stance_duration": 0.25,
    "total_duration": 0.5,
    "init_motor_pos": [0.0041, 0.9163, 0.4386, -1.12, 0.6975, -0.0041, -0.9163, 0.4386, -1.12, 0.6975],
    "use_footstep_plan": True,
}

from mujoco_env.cassie_env import CassieEnv
env = CassieEnv(config)

obs = env.reset()
done = False
steps = 0
falls = 0

print("Initial reset complete. Starting simulation loop...")

# Run simulation using random actions or zero actions to see if it terminates correctly
for _ in range(500):
    acs = np.zeros(10) # zero actions
    obs, obs_pol, reward, done, info = env.step(acs)
    steps += 1
    if done:
        if env.fall_flag:
            falls += 1
            print(f"Fell down at step {steps}")
        else:
            print(f"Episode completed successfully at step {steps}")
        break

print(f"Total steps: {steps}")
print(f"Is footstep plan used: {env.reference_generator.use_footstep_plan}")
print(f"Footstep traj length: {len(env.reference_generator.footstep_traj)}")
print(f"Final footstep index: {env.reference_generator.footstep_index}")
