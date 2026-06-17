import tensorflow as tf
from mujoco_env.cassie_env import CassieEnv
import ppo.policies as policies
from configs.env_config import config_play
from configs.defaults import ROOT_PATH
import argparse
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config_play["is_visual"] = False
# We will use smooth midpoints trajectory to see if the old checkpoint can follow it!
env = CassieEnv(config=config_play)

# Monkey patch interpolate_trajectory to smooth it
import math
def smooth_interpolate(footsteps, dt=0.03, walking_speed=0.5):
    if len(footsteps) < 2: return footsteps
    base_waypoints = [(0.0, 0.0, footsteps[0][2])]
    for i in range(len(footsteps) - 1):
        x = (footsteps[i][0] + footsteps[i+1][0]) / 2.0
        y = (footsteps[i][1] + footsteps[i+1][1]) / 2.0
        theta_diff = (footsteps[i+1][2] - footsteps[i][2] + math.pi) % (2 * math.pi) - math.pi
        theta = footsteps[i][2] + theta_diff / 2.0
        base_waypoints.append((x, y, theta))
    ref_traj = []
    for i in range(len(base_waypoints) - 1):
        x0, y0, th0 = base_waypoints[i]
        x1, y1, th1 = base_waypoints[i+1]
        dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
        steps = max(1, int((dist/walking_speed)/dt))
        for t in range(steps):
            alpha = t/steps
            x = (1-alpha)*x0 + alpha*x1
            y = (1-alpha)*y0 + alpha*y1
            t_diff = (th1-th0+math.pi)%(2*math.pi)-math.pi
            ref_traj.append((x, y, th0 + alpha*t_diff))
    ref_traj.append(base_waypoints[-1])
    return ref_traj

env.reference_generator.footstep_traj = smooth_interpolate(env.reference_generator.footstep_traj, dt=0.033)

model_dir = ROOT_PATH + "/ckpts/footstep_training_rnds42_cont3"
latest_checkpoint = tf.train.latest_checkpoint(model_dir)

config = tf.ConfigProto(device_count={"GPU": 0})
pi = policies.MLPCNNPolicy(
    name="pi", ob_space_vf=env.observation_space_vf, ob_space_pol=env.observation_space_pol,
    ob_space_pol_cnn=env.observation_space_pol_cnn, ac_space=env.action_space,
    hid_size=512, num_hid_layers=2,
)
import baselines.common.tf_util as U
U.make_session(config=config)
U.load_state(latest_checkpoint)

obs_vf, obs_pol = env.reset()
falls = 0
for step in range(200):
    ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
    obs_vf, obs_pol, reward, done, info = env.step(ac)
    if env.fall_flag:
        falls += 1
        print(f"Fell down at step {step}!")
        break
    if step % 20 == 0:
        print(f"Step {step}: robot pos X = {env.qpos[0]:.3f}, target X = {env.ref_dict['base_pos_global'][0]:.3f}")

if falls == 0:
    print("Old checkpoint successfully walked 200 steps on smooth trajectory without falling!")

