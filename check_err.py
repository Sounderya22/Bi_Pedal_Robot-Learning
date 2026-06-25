import tensorflow as tf
from mujoco_env.cassie_env import CassieEnv
import ppo.policies as policies
from configs.env_config import config_play
from configs.defaults import ROOT_PATH
import argparse
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = config_play.copy()
config["is_visual"] = False
config["use_footstep_plan"] = True

env = CassieEnv(config=config)

model_dir = ROOT_PATH + "/ckpts/footstep_training_v6_rnds42"
latest_checkpoint = tf.train.latest_checkpoint(model_dir)

config_tf = tf.ConfigProto(device_count={"GPU": 0})
pi = policies.MLPCNNPolicy(
    name="pi", ob_space_vf=env.observation_space_vf, ob_space_pol=env.observation_space_pol,
    ob_space_pol_cnn=env.observation_space_pol_cnn, ac_space=env.action_space,
    hid_size=512, num_hid_layers=2,
)
import baselines.common.tf_util as U
U.make_session(config=config_tf)
U.load_state(latest_checkpoint)

obs_vf, obs_pol = env.reset()
falls = 0
for step in range(50):
    ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
    obs_vf, obs_pol, reward, done, info = env.step(ac)
    
    pos_err_local = obs_pol[0][-2:]
    height = env.qpos[2]
    
    print(f"Step {step}: height={height:.3f}, pos_err_local={pos_err_local}")

    if env.fall_flag:
        falls += 1
        print(f"Fell down at step {step}! height={height:.3f}")
        break

if falls == 0:
    print("Agent survived.")

