import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from deep_rl_for_swarms.ma_envs.commons.utils import EzPickle
from environments.base import WildfireWorld, FloodWorld
from environments.agents.point_agents.aircraft_agent import Aircraft
import matplotlib.pyplot as plt
import cv2 as cv
import datetime


class SurveillanceEnv(gym.Env, EzPickle):

    """
    SurveillanceEnv class. This is a gym compatible environment that models a multiagent surveillanve scenario.
    It is based on and makes use of deep_rl_for_swarms: https://github.com/LCAS/deep_rl_for_swarms

    :param shared_reward: (bool) Whether all agents share the same average reward instead of using their own.
    """

    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self, nr_agents=5,
                 obs_type='wildfire',
                 obs_mode='normal',
                 obs_radius=500,
                 comm_radius=np.inf,
                 world_size=1000,
                 grid_size=100,
                 range_cutpoints=20,
                 angular_cutpoints=40,
                 torus=False,
                 dynamics='aircraft',
                 shared_reward=False,
                 render_dir='/tmp/video/'):
        EzPickle.__init__(self, nr_agents, obs_type, obs_mode, obs_radius, comm_radius, world_size, grid_size,
                          range_cutpoints,
                          angular_cutpoints, torus, dynamics)
        self.nr_agents = nr_agents
        self.world_size = world_size
        self.obs_type = obs_type
        self.obs_mode = obs_mode
        self.obs_radius = obs_radius
        if obs_type == 'wildfire':
            self.world = WildfireWorld(world_size, grid_size, torus, dynamics)
        elif obs_type == 'flood':
            self.world = FloodWorld(world_size, grid_size, torus, dynamics)
        else:
            raise Exception('Unknown obs_type')
        self.range_cutpoints = range_cutpoints
        self.angular_cutpoints = angular_cutpoints
        self.grid_size = grid_size
        self.torus = torus
        self.dynamics = dynamics
        self.comm_radius = comm_radius
        self.reward_mech = 'global'
        self.render_dir = render_dir
        self.hist = None
        self.world.agents = [
            Aircraft(self) for _ in
            range(self.nr_agents)
        ]
        self.shared_reward = shared_reward

        self.vel_hist = []
        self.state_hist = []
        self.timestep = 0
        self.ax = None
        self.fig = None
        self.top_fig = None
        self.top_ax = None

        # Reward parameters
        self.lambda1 = 1  # minimum distance to fire maximum penalty
        self.lambda2 = 2e-1  # inactive nearby cells maximum penalty
        self.lambda3 = 5e-2  # bank angle maximum penalty
        self.lambda4 = 1e-2  # distance to other aircraft maximum penalty
        self.r0 = 5  # inactive nearby cells radius to compute penalty
        self.c = 20  # distance to other aircraft where penalty drops to 0.36 of maximum

        # Position history
        self.pos_hist = None

        # Create render directory
        import os
        if not os.path.exists(render_dir):
            os.makedirs(render_dir)

    @property
    def state_space(self):
        """
        State space function.
        """

        return spaces.Box(low=-10., high=10., shape=(self.nr_agents * 3,), dtype=np.float32)

    @property
    def observation_space(self):
        """
        Observation space function.
        """

        return self.agents[0].observation_space

    @property
    def action_space(self):
        """
        Action space function
        """

        return self.agents[0].action_space

    @property
    def agents(self):
        """
        Agents function
        """

        return self.world.policy_agents

    def seed(self, seed=None):
        """
        Seed function
        """

        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def timestep_limit(self):
        """
        Timestep limit function
        """

        if self.obs_type == 'wildfire':
            return 5000
        elif self.obs_type == 'flood':
            return 10000

    @property
    def is_terminal(self):
        """
        Is terminal function
        """

        if self.timestep >= self.timestep_limit:
            return True
        else:
            return False

    def get_param_values(self):
        return self.__dict__

    @staticmethod
    def create_circle_mask(x_arr, y_arr, center, radius):
        """
        Creates a circle mask for reward calculation.
        """

        c_x, c_y = center
        dists_sqrd = (x_arr - c_x) ** 2 + (y_arr - c_y) ** 2
        return dists_sqrd <= radius ** 2

    def find_nearest(self, img, target):
        """
        Finds the nearest point of interest.
        """

        nonzero = cv.findNonZero(img.astype(int))
        if nonzero is None:
            return target + self.grid_size
        else:
            distances = np.sqrt((nonzero[:, :, 0] - target[0]) ** 2 + (nonzero[:, :, 1] - target[1]) ** 2)
            nearest_index = np.argmin(distances)
            return nonzero[nearest_index]

    def reset(self):
        """
        Reset function
        """

        self.timestep = 0

        agent_states = np.random.rand(self.nr_agents, 4)
        margin = 0.2
        agent_states[:, 0:2] = self.world_size * ((1 - 2 * margin) * agent_states[:, 0:2] + margin)  # Position
        agent_states[:, 2:3] = 2 * np.pi * agent_states[:, 2:3]  # Orientation
        agent_states[:, 3:4] = agent_states[:, 3:4] * 0  # Bank angle (always start at 0)

        self.world.agent_states = agent_states

        agent_list = [
            Aircraft(self)
            for _ in
            range(self.nr_agents)
        ]
        self.world.agents = agent_list

        self.world.reset()

        # Position history
        pos_hist = np.empty([self.timestep_limit + 1, self.nr_agents, 2])
        pos_hist[:] = np.nan

        pos_hist[self.timestep, :, :] = agent_states[:, 0:2]
        self.pos_hist = pos_hist

        obs = []

        for i, bot in enumerate(agent_list):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     self.world.agent_states[:, 3],
                                     self.world.active,
                                     self.world.agent_states[i, 0:2],
                                     self.world.agent_states[i, 2]
                                     )
            obs.append(ob)

        time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
        if self.obs_type == 'flood':
            self.render(render_type='relief', output_dir=self.render_dir, name_append=time)
        elif self.obs_type == 'wildfire':
            self.render(render_type='wildfire_state', output_dir=self.render_dir, name_append=time)

        return obs

    def step(self, actions):
        """
        Step function. Simulates one timestep for both the world and the agents.
        """

        self.timestep += 1

        for agent, action in zip(self.agents, actions):
            agent.action.u = action

        self.world.step()

        next_obs = []

        for i, bot in enumerate(self.agents):
            # 'image', 'own_bank_angle', 'range', 'rel_head_to_other', 'rel_head_from_other', 'other_bank_angle'
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     self.world.agent_states[:, 3],
                                     self.world.active,
                                     self.world.agent_states[i, 0:2],
                                     self.world.agent_states[i, 2]
                                     )
            next_obs.append(ob)

        rewards = []
        for bot in self.agents:
            rew_agent = self.get_reward(bot)
            rewards = np.append(rewards, rew_agent)

        if self.shared_reward:
            rewards = np.repeat(np.sum(rewards) / rewards.size, rewards.size)

        done = self.is_terminal

        self.pos_hist[self.timestep, :, :] = self.world.agent_states[:, 0:2]

        info = {'state': self.world.agent_states, 'actions': actions}

        return next_obs, rewards, done, info

    def get_reward(self, agent):
        """
        Get reward function. Computes the reward for the agent.
        """

        agent_grid_pos = np.round(agent.state.p_pos / self.world_size * self.grid_size)

        # Distance to closest point of interest in grid units
        nearest_active_pos = self.find_nearest(self.world.active, agent_grid_pos)
        dmin = np.sqrt(np.sum(np.square(nearest_active_pos - agent_grid_pos)))

        dmin = dmin * self.world_size / self.grid_size

        # Mask for nearby non-active cells penalty
        inactive_penalty_mask = self.create_circle_mask(*np.ogrid[0:self.grid_size, 0:self.grid_size],
                                                        center=agent_grid_pos, radius=self.r0)
        inactive_count = (~self.world.active & inactive_penalty_mask).sum()

        r1 = -self.lambda1 * dmin / self.world_size
        r2 = -self.lambda2 * inactive_count / (np.pi * self.r0 ** 2)
        r3 = -self.lambda3 * np.square(agent.state.p_bank_angle/agent.max_bank_angle)

        # Penalty for closeness to the other aircraft
        r4d = 0
        agents = self.world.agents
        for a_n in agents:
            if a_n != agent:
                r4d += np.sum(np.exp(-np.sqrt(np.sum(np.square(agent.state.p_pos - a_n.state.p_pos))) / self.c))
        r4 = -self.lambda4 * r4d

        r = r1 + r2 + r3 + r4
        return r

    def render(self, render_type='cells', mode='human', output_dir='/tmp/video/', name_append=''):
        """
        Render function. Renders the 2D representation of the environment.
        """

        if render_type == 'cells':
            if mode == 'animate':
                if self.timestep == 0:
                    import shutil
                    import os

                    os.makedirs(output_dir, exist_ok=True)

            if self.ax is None:
                self.fig, ax = plt.subplots(1, 2)
                ax0 = plt.subplot(121)
                ax1 = plt.subplot(122, projection='polar')
                self.ax = [ax0, ax1]

            plt.figure(self.fig.number)
            self.ax[0].clear()
            self.ax[0].set_aspect('equal')
            self.ax[0].set_xlim((0, self.world_size))
            self.ax[0].set_ylim((0, self.world_size))

            comm_circles = []
            self.ax[0].scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=20)
            arrow_max_length = self.world_size / 20
            for n in range(self.nr_agents):
                # Draw speed
                self.ax[0].arrow(self.world.agent_states[n, 0], self.world.agent_states[n, 1],
                                 arrow_max_length * np.cos(self.world.agent_states[n, 2]),
                                 arrow_max_length * np.sin(self.world.agent_states[n, 2]), color='g')
                # Draw bank angle
                self.ax[0].arrow(self.world.agent_states[n, 0], self.world.agent_states[n, 1],
                                 arrow_max_length * -np.sin(self.world.agent_states[n, 2]) * self.world.agent_states[
                                     n, 3],
                                 arrow_max_length * np.cos(self.world.agent_states[n, 2]) * self.world.agent_states[
                                     n, 3],
                                 color='r')
                # Draw trajectory history
                drawn_history_length = 1000
                time_start_traj = np.maximum(0, self.timestep - drawn_history_length)
                time_end_traj = self.timestep
                self.ax[0].plot(self.pos_hist[time_start_traj:time_end_traj, n, 0],
                                self.pos_hist[time_start_traj:time_end_traj, n, 1],
                                linewidth=1)
            for i in range(self.nr_agents):
                comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                                self.world.agent_states[i, 1]),
                                               self.comm_radius, color='g' if i != 0 else 'b', fill=False))

                self.ax[0].add_artist(comm_circles[i])

            active_plot = self.world.active
            self.ax[0].imshow(active_plot, extent=(0, self.world_size, 0, self.world_size))
            if mode == 'human':
                plt.pause(0.01)
            elif mode == 'animate':
                plt.savefig(output_dir + name_append + format(self.timestep, '04d'))

        if render_type == 'cells_v2':

            ax = plt.subplot()
            ax.clear()
            ax.imshow(self.world.active, extent=(0, self.world_size, 0, self.world_size), cmap='Blues')
            ax.scatter(self.world.agent_states[0, 0], self.world.agent_states[0, 1], c='g', s=20)

            arrow_max_length = self.world_size / 20
            for n in range(self.nr_agents):
                # Draw speed
                ax.arrow(self.world.agent_states[n, 0], self.world.agent_states[n, 1],
                                 arrow_max_length * np.cos(self.world.agent_states[n, 2]),
                                 arrow_max_length * np.sin(self.world.agent_states[n, 2]), color='r')
                # Draw bank angle
                ax.arrow(self.world.agent_states[n, 0], self.world.agent_states[n, 1],
                                 arrow_max_length * -np.sin(self.world.agent_states[n, 2]) * self.world.agent_states[
                                     n, 3],
                                 arrow_max_length * np.cos(self.world.agent_states[n, 2]) * self.world.agent_states[
                                     n, 3],
                                 color='r')
                # Draw trajectory history
                drawn_history_length = 1000
                time_start_traj = np.maximum(0, self.timestep - drawn_history_length)
                time_end_traj = self.timestep
                ax.plot(self.pos_hist[time_start_traj:time_end_traj, n, 0],
                                self.pos_hist[time_start_traj:time_end_traj, n, 1],
                                linewidth=1, color='o')

            ax.set_aspect('equal')
            ax.set_xlim((0, self.world_size))
            ax.set_ylim((0, self.world_size))

            plt.savefig(output_dir + name_append + format(self.timestep, '04d'))

        if render_type == 'relief':
            from landlab.plot import imshow_grid
            if self.top_fig is None:
                self.top_fig, ax = plt.subplots()
            else:
                plt.figure(self.top_fig.number)
            imshow_grid(self.world.mg, 'topographic__elevation', output=output_dir + name_append + '0-relief',
                        colorbar_label='m')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')

        if render_type == 'flood' or render_type == 'mapandobs':
            from landlab.plot import imshow_grid
            plt.figure()
            imshow_grid(self.world.mg, 'surface_water__depth', cmap='Blues',
                        output=output_dir + name_append + '1_flood_' + format(self.timestep, '04d'),
                        colorbar_label='Surface water depth [m]', limits=(0.0, 0.2))
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')

        if render_type == 'mapandobs':
            ax = plt.subplot()
            ax.imshow(self.world.active, extent=(0, self.world_size, 0, self.world_size), cmap='Blues')
            arrow_max_length = self.world_size / 20
            ax.arrow(self.world.agent_states[0, 0], self.world.agent_states[0, 1],
                     arrow_max_length * -np.sin(self.world.agent_states[0, 2]) * self.world.agent_states[0, 3],
                     arrow_max_length * np.cos(self.world.agent_states[0, 2]) * self.world.agent_states[0, 3],
                     color='r')
            ax.scatter(self.world.agent_states[0, 0], self.world.agent_states[0, 1], c='g', s=20)
            plt.savefig(output_dir + name_append + '2_visible_' + format(self.timestep, '04d'))

        if render_type == 'wildfire_state':
            if self.top_fig is None:
                self.top_fig, self.top_ax = plt.subplots(1, 2)
                ax_fuel = plt.subplot(121)
                ax_fire = plt.subplot(122)
                self.top_ax = [ax_fuel, ax_fire]
            else:
                plt.figure(self.top_fig.number)

            self.top_ax[0].clear()
            self.top_ax[0].set_aspect('equal')
            self.top_ax[0].set_xlim((0, self.world_size))
            self.top_ax[0].set_ylim((0, self.world_size))
            self.top_ax[0].imshow(self.world.active, extent=(0, self.world_size, 0, self.world_size))

            self.top_ax[1].clear()
            self.top_ax[1].set_aspect('equal')
            self.top_ax[1].set_xlim((0, self.world_size))
            self.top_ax[1].set_ylim((0, self.world_size))
            self.top_ax[1].imshow(self.world.fuel, extent=(0, self.world_size, 0, self.world_size))

            plt.savefig(output_dir + name_append + 'wildfire_state_' + format(self.timestep, '04d'))