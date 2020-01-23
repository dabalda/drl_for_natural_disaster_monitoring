import os
import datetime
from environments.envs.point_envs.surveillance import SurveillanceEnv
import numpy as np

def main():
    log_dir = '../../drl_for_surveillance_runs/wildfire/indiv/dqn/report/'
    n_agents = 2
    env = SurveillanceEnv(nr_agents=n_agents ,
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

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        timestep = 0
        while True:
            if timestep % 500 == 0:
                time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
                env.render(render_type='wildfire_state',
                           output_dir=log_dir + 'video/',
                           name_append=time + format(timestep, '04d') + '_')

            action = np.zeros(n_agents)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
            timestep += 1
        print("Episode reward", episode_rew)




if __name__ == '__main__':
    main()