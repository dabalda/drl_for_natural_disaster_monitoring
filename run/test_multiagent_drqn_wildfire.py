#!/usr/bin/env python3
from algorithms.ma_drqn import DRQN
from policies import StatePlusImageLstmPolicy
from environments.envs.point_envs.surveillance import SurveillanceEnv
from callbacks.surveillance_callback import callback
from run.util import train


def main():
    log_dir = '../../drl_for_surveillance_runs/wildfire/indiv/drqn/test/'

    env = SurveillanceEnv(nr_agents=2,
                          obs_mode='normal',
                          obs_type='wildfire',
                          obs_radius=500,
                          world_size=1000,
                          grid_size=100,
                          range_cutpoints=30,
                          angular_cutpoints=40,
                          torus=False,
                          dynamics='aircraft',
                          shared_reward=False,
                          render_dir=log_dir + 'video/')

    policy = StatePlusImageLstmPolicy

    model = DRQN(policy, env, prioritized_replay=False,
                 verbose=1,
                 tensorboard_log=log_dir,
                 batch_size=100,
                 target_network_update_freq=100,
                 learning_starts=10000,
                 train_freq=10,
                 trace_length=20,
                 h_size=100,
                 learning_rate=5e-4,
                 buffer_size=200000,
                 exploration_fraction=0.7,
                 exploration_final_eps=0.1,
                 checkpoint_path=log_dir + 'models/',
                 checkpoint_freq=10000)

    params_val = model.sess.run(model.graph._collections['trainable_variables'][22])
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(params_val.flatten(), 40, alpha=0.3)

    plt.show()
    print('')

if __name__ == '__main__':
    main()
