#!/usr/bin/env python3
from algorithms.ma_dqn import DQN
from policies import StatePlusImageLstmPolicy
from environments.envs.point_envs.surveillance import SurveillanceEnv
from callbacks.surveillance_callback import callback
from run.util import train


def main():
    log_dir = '../../drl_for_surveillance_runs/'
    saved_model_name = 'checkpoint_model.pkl'

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

    model = DQN.load(log_dir + 'models/' + saved_model_name, env=env)

    obs = env.reset()
    states = None
    while True:
        actions, states = model.predict(obs, states=states)
        obs, rewards, dones, info = env.step(actions)
        if env.timestep % 100 == 0:
            env.render()
        if dones:
            obs = env.reset()
            states = None


if __name__ == '__main__':
    main()
