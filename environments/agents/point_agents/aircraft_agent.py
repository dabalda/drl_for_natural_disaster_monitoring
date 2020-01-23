from deep_rl_for_swarms.ma_envs.base import Agent
from gym import core, spaces, wrappers
import numpy as np


class Aircraft(Agent):
    """
    Aircraft class. It models a deep_rl_for_swarms agent consisting of a fixed wing steerable aircraft with
    constant velocity.

    :param experiment: (SurveillanceEnv) the experiment environment.
    """

    def __init__(self, experiment):
        super(Aircraft, self).__init__()
        self.comm_radius = experiment.comm_radius
        self.obs_radius = experiment.obs_radius
        self.obs_mode = experiment.obs_mode
        self.torus = experiment.torus
        self.n_agents = experiment.nr_agents
        self.world_size = experiment.world_size
        self.grid_size = experiment.grid_size
        self._dim_a = 1
        self.range_cutpoints = experiment.range_cutpoints
        self.angular_cutpoints = experiment.angular_cutpoints
        self.max_bank_angle = 50 * np.pi / 180  # (rad)

        # Aircraft dynamics
        self.state.p_bank_angle = None  # (rad)
        self.bank_angle_step = 5 * np.pi / 180  # (rad)
        self.constant_vel = 20  # (m/s)
        self.velocity = None  # (m/s)

        self.obs_angles = np.linspace(0, 2 * np.pi, num=self.angular_cutpoints, endpoint=False)
        self.obs_range_ang_delta = 2.635  # degrees
        obs_ranges = np.tan(np.arange(1, self.range_cutpoints + 1) * np.deg2rad(self.obs_range_ang_delta))
        self.obs_ranges = np.array(obs_ranges / obs_ranges[-1] * self.obs_radius)

        if self.obs_mode == 'normal':
            self.dim_rec_o = (self.range_cutpoints, self.angular_cutpoints)
            self.dim_local_o = 1
            self.n_obs_others = 4
            self.dim_obs_others = experiment.nr_agents - 1

        else:
            raise ValueError('obs mode must be 1D or 2D')

    @property
    def observation_space(self):
        """
        Observation space function
        """

        ob_space = spaces.Box(low=-np.inf, high=np.inf,
                              shape=(self.range_cutpoints * self.angular_cutpoints + 1 + (self.n_agents - 1) * 4,),
                              dtype=np.float32)

        ob_space.dim_image = (self.range_cutpoints, self.angular_cutpoints)
        ob_space.dim_state = self.n_agents - 1
        return ob_space

    @property
    def action_space(self):
        """
        Action space function
        """

        return spaces.Discrete(2)

    def set_velocity(self, vel):
        """
        Set velocity function

        :param vel: (float) the velocity.
        """

        self.velocity = vel

    def reset(self, state):
        """
        Reset function

        :param state: (AgentState) the state to reset the agent to.
        """

        self.state.p_pos = state[0:2]
        self.state.p_orientation = state[2]
        self.state.p_bank_angle = state[3]

    def get_observation(self, dm, my_orientation, their_orientation, bank_angle, detection_map, position, heading):
        """
        Get observation function

        :param dm: (array) distance matrix.
        :param my_orientation: (array) relative orientation of this aircraft with respect to the position of the other aircraft.
        :param their_orientation: (array) relative orientation of the other aircraft with respect to this aircraft.
        :param bank_angle: (float) bank angle of this aircraft.
        :param detection_map: (array) map of points of interest.
        :param position: (array) position of the aircraft in the map.
        :param heading: (float) absolute orientation of this aircraft.
        """

        if self.obs_mode == 'normal':
            ind = np.where(dm == -1)[0][0]
            own_bank_angle = self.state.p_bank_angle / self.max_bank_angle  # Normalized own bank angle
            others_obs = np.zeros([self.n_agents - 1, 4])

            # Rotate sample points according to heading
            rotated_angle = self.obs_angles + heading + np.pi
            ranges_mat, rotated_angle_mat = np.meshgrid(self.obs_ranges, rotated_angle)

            d_x = ranges_mat * np.cos(rotated_angle_mat)
            d_y = ranges_mat * np.sin(rotated_angle_mat)

            x_map = np.around((d_x + position[0]) * self.grid_size / self.world_size)
            y_map = np.around((d_y + position[1]) * self.grid_size / self.world_size)

            np.clip(x_map, 0, self.grid_size - 1, out=x_map)
            np.clip(y_map, 0, self.grid_size - 1, out=y_map)

            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

            image = detection_map[x_map, y_map]

            others_obs[:, 0] = np.concatenate([dm[0:ind], dm[ind + 1:]]) / (self.world_size / 2)  # normalized range
            others_obs[:, 1] = np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]) / (
                        2 * np.pi)  # normalized rel_head_to_other
            others_obs[:, 2] = np.concatenate(
                [their_orientation[0:ind], their_orientation[ind + 1:]]) / (2 * np.pi)  # normalized rel_head_from_other
            others_obs[:, 3] = np.concatenate(
                [bank_angle[0:ind], bank_angle[ind + 1:]]) / self.max_bank_angle  # normalized other_bank_angle

            obs = np.hstack([image.flatten(order='F'), own_bank_angle, others_obs.flatten()])
        else:
            raise ValueError('Non-existing observation mode')
        return obs
