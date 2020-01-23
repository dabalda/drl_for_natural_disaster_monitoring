#!/usr/bin/env python3
from algorithms.ma_dqn import DQN
from environments.envs.point_envs.surveillance import SurveillanceEnv
import numpy as np
import datetime

def main():
    log_dir = '../../../models/wildfire/indiv/dqn/'
    saved_model_name = 'checkpoint_model.pkl'
    n_agents = 2
    env = SurveillanceEnv(nr_agents=n_agents,
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

    # All tests share the same seed
    env.seed(seed=1)
    np.random.seed(seed=1)

    model = DQN.load(log_dir + 'models/' + saved_model_name, env=env)

    total_episodes = 100

    episode_rewards = []
    for ep in range(total_episodes):

        obs = env.reset()
        states = None
        episode_rewards.append(0.0)

        while True:
            actions, states = model.predict(obs, states=states)
            obs, rewards, dones, info = env.step(actions)

            for i in range(n_agents):
                # Sum of rewards for logging
                episode_rewards[-1] += rewards[i]

            if env.timestep % 500 == 0:
                print("Timestep " + str(env.timestep) + "/" + str(env.timestep_limit))

            algo = model
            time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

            # Render map
            if (algo.env.is_terminal or algo.env.timestep % 500 == 0) \
                    and algo.env.timestep > 0:

                n = 1  # agent to plot
                dim_image = algo.observation_space.dim_image
                obs_array = np.array(obs[n])
                obs_image = np.reshape(obs_array[:dim_image[0] * dim_image[1]], dim_image, order='C')

                if algo.env.ax is not None:
                    x = algo.env.agents[n].obs_angles
                    y = algo.env.agents[n].obs_ranges
                    ax1 = algo.env.ax[1]
                    ax1.set_theta_zero_location("S")

                    x_mod = np.append(x, 2 * np.pi)
                    obs_image_append = np.expand_dims(obs_image[:, 0], axis=1)
                    obs_image_mod = np.append(obs_image, obs_image_append, axis=1)

                    ax1.pcolormesh(x_mod, y, obs_image_mod)

                render_dir = algo.env.render_dir
                algo.env.render(mode='animate',
                                output_dir=render_dir + time + '-',
                                name_append=format(algo.num_timesteps, '04d') + '_')
                print(time)
                print("Instantaneous rewards:", rewards)

            if dones:
                print("Episode", ep+1, "reward:", episode_rewards[-1])
                break

        np.save(log_dir + 'test_' + str(total_episodes) + '_episode_rewards', episode_rewards)


if __name__ == '__main__':
    main()
