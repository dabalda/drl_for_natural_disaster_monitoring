#!/usr/bin/env python3
from algorithms.ma_drqn import DRQN
from environments.envs.point_envs.surveillance import SurveillanceEnv
from stable_baselines.a2c.utils import find_trainable_variables
import tensorflow as tf
from tensorflow.python import debug as tf_debug


def main():
    log_dir = '../../../../Results/models/wildfire/indiv/drqn/obs_100/'
    saved_model_name = 'checkpoint_model.pkl'

    config = tf.ConfigProto(log_device_placement=True)

    env = SurveillanceEnv(nr_agents=2,
                          obs_mode='normal',
                          obs_type='wildfire',
                          obs_radius=100,  # CHECK
                          world_size=1000,
                          grid_size=100,
                          range_cutpoints=30,
                          angular_cutpoints=40,
                          torus=False,
                          dynamics='aircraft',
                          shared_reward=False,
                          render_dir=log_dir + 'video/')

    model = DRQN.load(log_dir + 'models/' + saved_model_name, env=env)
    params = find_trainable_variables("main")

    params_val = model.sess.run(model.graph._collections['trainable_variables'][6])
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(params_val.flatten(), 40, alpha=0.3)

    plt.show()
    print('')

if __name__ == '__main__':
    main()
