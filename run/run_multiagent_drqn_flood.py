#!/usr/bin/env python3
from algorithms.ma_drqn import DRQN
from policies import StatePlusImageLstmPolicy
from environments.envs.point_envs.surveillance import SurveillanceEnv
from callbacks.surveillance_callback import callback
from run.util import train


def main():
    log_dir = '../../drl_for_surveillance_runs/flood/indiv/drqn/v1/'

    env = SurveillanceEnv(nr_agents=2,
                          obs_mode='normal',
                          obs_type='flood',
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
                 learning_starts=15000,
                 train_freq=10,
                 trace_length=20,
                 buffer_size=200000,
                 exploration_fraction=0.7,
                 exploration_final_eps=0.1,
                 checkpoint_path=log_dir + 'models/',
                 checkpoint_freq=10000)

    train(model, callback, num_timesteps=int(3e6), log_dir=log_dir)


if __name__ == '__main__':
    main()
