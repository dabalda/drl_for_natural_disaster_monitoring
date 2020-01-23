import datetime
import numpy as np
import matplotlib.pyplot as plt

start_time = datetime.datetime.now()
fig_rew, ax_rew = plt.subplots(1, 1)


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN and others) or after n steps

    :param _locals: (dict)
    :param _globals: (dict)
    """

    algo = _locals['self']
    time = datetime.datetime.now()
    time_format_print = "%Y-%m-%d %H:%M:%S"

    # Render map
    if (algo.env.is_terminal or algo.env.timestep % 500 == 0) \
            and algo.env.timestep > 0:

        n = 1  # agent to plot
        dim_image = algo.observation_space.dim_image
        obs = np.array(_locals['new_obs'][n])
        obs_image = np.reshape(obs[:dim_image[0] * dim_image[1]], dim_image, order='C')

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
                        output_dir=render_dir + time.strftime("%Y_%m_%d-%H_%M_%S") + '-',
                        name_append=format(algo.num_timesteps, '04d') + '_')
        print(time.strftime(time_format_print))
        print("Step:", algo.num_timesteps)
        print("Instantaneous rewards:", _locals['rew'])
        diff_time = time - start_time
        diff_time_sec = diff_time.total_seconds()
        est_time_left = diff_time_sec / algo.num_timesteps * _locals['total_timesteps']
        est_finish_time = time + datetime.timedelta(seconds=est_time_left)
        print("Estimated finish time:", est_finish_time.strftime(time_format_print))
        print("Estimated time left:", str(datetime.timedelta(seconds=est_time_left)))

    # Save model
    if algo.checkpoint_path is not None and \
            algo.num_timesteps > algo.learning_starts and \
            algo.num_timesteps % algo.checkpoint_freq == 0:
        algo.save(algo.checkpoint_path + time.strftime("%Y_%m_%d-%H_%M_%S") + '-' + 'checkpoint_model')

    if _locals['reset'] and 'num_episodes' in _locals.keys() and _locals['num_episodes'] > 1:
        print("Episode", _locals['num_episodes']-1, "reward:", _locals['episode_rewards'][-2])
        print("Mean reward per episode: " + str(np.average(_locals['episode_rewards'][:-1])))
        print("Best episode reward so far: " + str(max(_locals['episode_rewards'][:-1])) +
              ', episode ' + str(np.argmax(_locals['episode_rewards'][:-1])+1))

        plt.figure(fig_rew.number)
        ax_rew.clear()
        ax_rew.plot(_locals['episode_rewards'][:-1])
        filename = algo.tensorboard_log + 'reward_plots/' + start_time.strftime("%Y_%m_%d-%H_%M_%S") + '_episode_reward'
        plt.savefig(filename)
        np.save(filename, _locals['episode_rewards'][:-1])

    return True
