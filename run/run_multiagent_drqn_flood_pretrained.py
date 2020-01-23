#!/usr/bin/env python3
from algorithms.ma_drqn import DRQN
from policies import StatePlusImageLstmPolicy
from environments.envs.point_envs.surveillance import SurveillanceEnv
from callbacks.surveillance_callback import callback
from run.util import train


def main():
    log_dir = '../../drl_for_surveillance_runs/flood/indiv/drqn/pretrained/'
    pretrained_model_name = 'drqn-20_model.pkl'

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

    model = DRQN.load('../../' + 'pretrained_models/' + pretrained_model_name, env=env)

    model.tensorboard_log = log_dir
    model.checkpoint_path = log_dir + 'models/'
    model.initial_exploration = 0.5

    train(model, callback, num_timesteps=int(2e6), log_dir=log_dir)


if __name__ == '__main__':
    main()
