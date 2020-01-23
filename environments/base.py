import numpy as np
import deep_rl_for_swarms.ma_envs.commons.utils as U
from scipy import signal
from landlab.components.overland_flow import OverlandFlow
from landlab.components import FlowAccumulator, FastscapeEroder
from landlab import RasterModelGrid
import terrain_erosion_3_ways.util as te3w_util

dynamics = ['point', 'unicycle', 'box2d', 'direct', 'unicycle_acc', 'aircraft']


class World(object):
    """
    World class. It models the physical environment in which the agents are placed.

    :param world_size:
    :param torus:
    :param agent_dynamic:
    """

    def __init__(self, world_size, torus, agent_dynamic):
        self.nr_agents = None
        # world is square
        self.world_size = world_size
        # dynamics of agents
        assert agent_dynamic in dynamics
        self.agent_dynamic = agent_dynamic
        # periodic or closed world
        self.torus = torus
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # matrix containing agent states
        self.agent_states = None
        # matrix containing landmark states
        self.landmark_states = None
        # x,y of everything
        self.nodes = None
        self.distance_matrix = None
        self.angle_matrix = None
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1  # (s)
        self.action_repeat = 1
        self.timestep = 0
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        self.g = 9.8  # gravity (m/s^2)

    @property
    def entities(self):
        """
        Entities function. Returns all entities in the world.
        """

        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        """
        Policy agents function. Returns all agents controllable by external policies.
        """

        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        """
        Scripted agents function. Returns all agents controlled by world scripts.
        """

        return [agent for agent in self.agents if agent.action_callback is not None]

    def reset(self):
        """
        Reset function. Resets the world to its initial state.
        """

        self.timestep = 0
        self.nr_agents = len(self.policy_agents)

        for i, agent in enumerate(self.policy_agents):
            agent.reset(self.agent_states[i, :])

        self.nodes = self.agent_states[:, 0:2]

        self.distance_matrix = U.get_distance_matrix(self.nodes,
                                                     torus=self.torus, world_size=self.world_size, add_to_diagonal=-1)

        angles = np.vstack([U.get_angle(self.nodes, a[0:2],
                                        torus=self.torus, world_size=self.world_size) - a[2] for a in
                            self.agent_states])
        angles_shift = -angles % (2 * np.pi)
        self.angle_matrix = np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)

    def step(self):
        """
        Step function. Simulates one timestep in the world.
        """

        self.timestep += 1

        for i, agent in enumerate(self.scripted_agents):
            action = agent.action_callback(agent, self)
            if agent.dynamics == 'direct':
                next_coord = agent.state.p_pos + action * agent.max_speed * self.dt * self.action_repeat
                if self.torus:
                    next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
                    next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
                else:
                    next_coord = np.where(next_coord < 0, 0, next_coord)
                    next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)
                agent.state.p_pos = next_coord

            self.landmark_states[i, :] = agent.state.p_pos

        if self.agent_dynamic == 'aircraft':
            # aircraft dynamics: velocity is constant, bearing rate of change depends on bank angle

            scaled_actions = np.zeros([self.nr_agents, 1])
            for i, agent in enumerate(self.policy_agents):
                assert agent.action.u in [0, 1]
                if agent.action.u == 0:
                    scaled_actions[i] = agent.bank_angle_step
                else:
                    scaled_actions[i] = -agent.bank_angle_step
            agent_states_next = np.copy(self.agent_states)  # pos x, pos y, orientation, bank angle

            # Old bank angle
            bank_angles = np.vstack([agent.state.p_bank_angle for agent in self.policy_agents])
            max_bank_angles = np.stack([agent.max_bank_angle for agent in self.policy_agents])

            # New bank angle
            bank_angles += scaled_actions

            # Clip new bank angle
            bank_angles[:, 0] = np.where(np.abs(bank_angles[:, 0]) > max_bank_angles,
                                         np.sign(bank_angles[:, 0]) * max_bank_angles,
                                         bank_angles[:, 0])

            # Constant velocity
            constant_vels = np.vstack([agent.constant_vel for agent in self.policy_agents])

            # Old orientation
            orientations = np.vstack([agent.state.p_orientation for agent in self.policy_agents])

            # Angular velocity
            ang_vel = self.g * np.tan(bank_angles) / constant_vels

            # New orientation
            orientations += ang_vel * self.dt

            # Wrap orientation to [0, 2*pi)
            orientations[:, 0] = np.where(orientations[:, 0] >= 2 * np.pi,
                                          orientations[:, 0] - 2 * np.pi,
                                          orientations[:, 0])
            orientations[:, 0] = np.where(orientations[:, 0] < 0,
                                          orientations[:, 0] + 2 * np.pi,
                                          orientations[:, 0])

            # New position
            step = np.concatenate([constant_vels * np.cos(orientations),
                                   constant_vels * np.sin(orientations)],
                                  axis=1)

            next_coord = agent_states_next[:, 0:2] + step * self.dt

            if self.torus:
                next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
                next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
            else:
                next_coord = np.where(next_coord < 0, 0, next_coord)
                next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)

            agent_states_next = np.concatenate([next_coord, orientations, bank_angles], axis=1)

            self.agent_states = agent_states_next

            for i, agent in enumerate(self.policy_agents):
                agent.state.p_pos = agent_states_next[i, 0:2]
                agent.state.p_orientation = agent_states_next[i, 2:3]
                agent.state.p_bank_angle = agent_states_next[i, 3:4]

        self.nodes = self.agent_states[:, 0:2]

        self.distance_matrix = U.get_distance_matrix(self.nodes,
                                                     torus=self.torus, world_size=self.world_size, add_to_diagonal=-1)

        angles = np.vstack([U.get_angle(self.nodes, a[0:2],
                                        torus=self.torus, world_size=self.world_size) - a[2] for a in
                            self.agent_states])
        angles_shift = -angles % (2 * np.pi)
        self.angle_matrix = np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)


class ActiveCellWorld(World):
    """
    ActiveCellWorld class

    :param world_size:
    :param grid_size:
    :param torus:
    :param agent_dynamic:
    """

    def __init__(self, world_size, grid_size, torus, agent_dynamic):
        super(ActiveCellWorld, self).__init__(world_size, torus, agent_dynamic)
        self.active = np.full((grid_size,) * 2, False)
        self.cell_update_period = None
        self.time_since_cell_update = 0
        self.grid_size = grid_size

    def step(self):
        """
        Step function. Simulates one timestep in the world.
        """

        super(ActiveCellWorld, self).step()

        self.time_since_cell_update += self.dt
        if self.time_since_cell_update >= self.cell_update_period:
            self.update_cells()
            self.time_since_cell_update = 0

    def update_cells(self):
        pass


class WildfireWorld(ActiveCellWorld):
    """
    WildfireWorld class

    :param world_size:
    :param grid_size:
    :param torus:
    :param agent_dynamic:
    """

    def __init__(self, world_size, grid_size, torus, agent_dynamic):
        super(WildfireWorld, self).__init__(world_size, grid_size, torus, agent_dynamic)
        self.cell_update_period = 2.5  # (s)
        self.wind = 0.02  # <1
        self.max_ignition_radius = 2
        self.m_log_p_not_ig = None
        self.fuel = np.zeros((self.grid_size,) * 2)
        self.burning_rate = 1
        self.default_fuel_low = 15
        self.default_fuel_high = 20

    def reset(self):
        """
        Reset function. Resets the world to its initial state.
        """

        super(WildfireWorld, self).reset()

        dim_ig_prob = 1+2*self.max_ignition_radius
        m_dist_ig = np.zeros((dim_ig_prob,) * 2)

        # Use logarithm of product
        for index, _ in np.ndenumerate(m_dist_ig):
                m_dist_ig[index] = np.sqrt(np.sum(np.square([index[0]-self.max_ignition_radius,
                                                             index[1]-self.max_ignition_radius])))
        m_dist_ig[self.max_ignition_radius, self.max_ignition_radius] = np.inf  # Avoid division by 0
        m_p_not_ig = 1 - self.wind * np.power(m_dist_ig, -2)  # Probability of not igniting matrix
        self.m_log_p_not_ig = np.log(m_p_not_ig)

        # Setup initial fuel
        self.fuel = np.random.randint(self.default_fuel_low, high=self.default_fuel_high,
                                      size=[self.grid_size, self.grid_size])

        # Setup initial wildfire
        self.active = np.full((self.grid_size, self.grid_size), False)
        x = np.round(self.grid_size / 2).astype(int)
        y = x
        x_spread = 1
        y_spread = 1
        self.active[x - x_spread:x + 1 + x_spread, y - y_spread:y + 1 + y_spread] = True

        # Run wildfire for 50 seconds before aircraft arrive
        initial_run_time = 50  # (s)
        n_update_periods = int(round(initial_run_time / self.cell_update_period))
        for step in range(n_update_periods):
            self.update_cells()

    def update_cells(self):
        """
        Update cells function. Updates wildfire fuel and burning status.
        """

        next_fuel = np.where(self.active, self.fuel - self.burning_rate, self.fuel)  # Consume fuel if burning
        next_fuel = np.clip(next_fuel, 0, None)  # Fuel cannot be negative

        # Sum logs of probabilities of not igniting
        burning_int = self.active.astype(int)
        log_not_ignite = signal.convolve2d(burning_int, self.m_log_p_not_ig, mode='same', fillvalue=0)
        p_not_ignite = np.power(np.e, log_not_ignite)  # Probability of not being ignited
        p_ignition = 1 - p_not_ignite  # Probability of being ignited

        # Sample from ignition probabilities
        ignition = np.random.random_sample([self.grid_size, self.grid_size]) < p_ignition

        next_burning = np.logical_or(self.active, ignition)  # Previously burning or just ignited
        next_burning = np.where(next_fuel != 0, next_burning, False)  # Cells with no fuel are no longer burning

        # Save
        self.active = next_burning
        self.fuel = next_fuel


class FloodWorld(ActiveCellWorld):
    """
    FloodWorld class. This ActiveCellWorld implementation uses Landlab to simulate floods.

    :param world_size:
    :param grid_size:
    :param torus:
    :param agent_dynamic:
    """

    def __init__(self, world_size, grid_size, torus, agent_dynamic):
        super(FloodWorld, self).__init__(world_size, grid_size, torus, agent_dynamic)

        # World generation parameters
        self.vertical_scale = 200  # (m)

        self.inputs = {'nrows': 100, 'ncols': 100, 'dx': 0.02, 'dt': 0.5, 'total_time': 50.0, 'uplift_rate': 0.001,
                       'K_sp': 0.3, 'm_sp': 0.5, 'n_sp': 1.0, 'rock_density': 2.7, 'sed_density': 2.7,
                       'linear_diffusivity': 0.0001}

        # Flood parameters
        self.h_init = 5  # initial thin layer of water (m)
        self.n = 0.01  # roughness coefficient, (s/m^(1/3))
        self.alpha = 0.7  # time-step factor (nondimensional; from Bates et al., 2010)
        self.u = 0.4  # constant velocity (m/s, de Almeida et al., 2012)

        self.cell_update_period = 10  # (s)

        self.mg = None
        self.fr = None
        self.sp = None
        self.of = None
        self.swd = None

        self.z = None
        self.flood_threshold = 0.05  # minimum water depth detected as flood (m)

    def reset(self):
        """
        Reset function. Resets the world to its initial state.
        """

        super(FloodWorld, self).reset()

        # Terrain generation
        shape = (self.grid_size,) * 2
        size = self.grid_size * self.grid_size
        # Set initial simple topography
        z = self.vertical_scale * self.simple_topography(shape)

        # Create raster model if it does not exist
        if self.mg is None:
            self.mg = RasterModelGrid(shape, int(round(self.world_size/self.grid_size)))
            self.mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
            self.z = self.mg.add_field('node', 'topographic__elevation', z)
            self.swd = self.mg.add_zeros('node', 'surface_water__depth')
        else:
            self.mg.at_node['topographic__elevation'] = z
            self.swd[:] = np.zeros(size)

        import matplotlib.pyplot as plt
        plt.figure()
        from landlab.plot import imshow_grid
        imshow_grid(self.mg, 'topographic__elevation',
                    colorbar_label='m')
        plt.draw()

        # Set evolution parameters
        uplift_rate = self.inputs['uplift_rate']
        total_t = self.inputs['total_time']
        dt = self.inputs['dt']

        nt = int(total_t // dt)  # Loops
        uplift_per_step = uplift_rate * dt

        self.fr = FlowAccumulator(self.mg, **self.inputs)
        self.sp = FastscapeEroder(self.mg, **self.inputs)

        # Erode terrain
        for i in range(nt):
            self.fr.run_one_step() # Not time sensitive
            self.sp.run_one_step(dt)
            self.mg.at_node['topographic__elevation'][self.mg.core_nodes] += uplift_per_step  # add the uplift
            if i % 10 == 0:
                print('Erode: Completed loop %d' % i)

        plt.figure()
        imshow_grid(self.mg, 'topographic__elevation',
                    colorbar_label='m')
        plt.draw()
        plt.show()

        # Setup surface water flow
        self.of = OverlandFlow(self.mg, steep_slopes=True, mannings_n=0.01)

        # Setup initial flood
        self.swd[[5050, 5051, 5150, 5151]] += self.h_init
        self.active = np.greater(np.flip(np.reshape(self.swd, shape), axis=0), self.flood_threshold)

    def update_cells(self):
        """
        Update cells function. Updates water level and cell status.
        """

        self.of.overland_flow(dt=self.cell_update_period)
        shape = (self.grid_size,) * 2
        self.active = np.greater(np.flip(np.reshape(self.swd, shape), axis=0), self.flood_threshold)

    @staticmethod
    def noise_octave(shape, f):
        """
        Noise octave function. Generates a noise map.

        :param shape: (array) shape of the map.
        :param f: (float) frequency bounds parameter.
        """

        return te3w_util.fbm(shape, -1, lower=f, upper=(2 * f))

    def simple_topography(self, shape):
        """
        Simple topography function. Generates an elevation map.

        :param shape: (array) shape of the map.
        """

        values = np.zeros(shape)
        for p in range(1, 10):
            a = 2 ** p
            values += np.abs(self.noise_octave(shape, a) - 0.5) / a
        result = (1.0 - te3w_util.normalize(values)) ** 2
        return result
