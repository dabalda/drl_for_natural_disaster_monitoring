import numpy as np
import tensorflow as tf
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.schedules import LinearSchedule
from .replay_buffer import ReplayBuffer
from .util import updateTargetGraph, updateTarget
from stable_baselines.a2c.utils import find_trainable_variables
import datetime


class DRQN(OffPolicyRLModel):
    """
    The DRQN model class. This is a multiagent implementation of the DRQN algorithm based on this project's DQN
    implementation.
    DRQN paper: https://arxiv.org/abs/1507.06527

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor on the target Q-values
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) how many experience traces to use for each training step.
    :param checkpoint_freq: (int) how often to save the model. This is so that the best version is restored at the
            end of the training. If you do not wish to restore the best version
            at the end of the training set this variable to None.
    :param checkpoint_path: (str) replacement path used if you need to log to somewhere else than a temporary
            directory.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float) alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param trace_length: (int) length of training experience traces
    :param h_size (int) the size of the final layer before splitting it into advantage and value streams.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, trace_length=20, h_size=200):

        super(DRQN, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, policy_base=None,
                                   requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.checkpoint_path = checkpoint_path
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.checkpoint_freq = checkpoint_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.trace_length = trace_length
        self.h_size = h_size

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.main_qnet = None
        self.target_qnet = None

        if hasattr(self.env, 'nr_agents'):
            self.n_agents = self.env.nr_agents
            self.magent = True
        else:
            self.n_agents = 1
            self.magent = False

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        pass

    def setup_model(self):
        """
        Model setup function
        """
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.main_qnet = self.policy(h_size=self.h_size,
                                             scope='main',
                                             env=self.env,
                                             optimizer=optimizer)
                self.target_qnet = self.policy(h_size=self.h_size,
                                               scope='target',
                                               env=self.env,
                                               optimizer=optimizer)

                self.params = find_trainable_variables("deeprq")

                tf_util.initialize(self.sess)

                self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DRQN",
              reset_num_timesteps=True):
        """
        Learn function
        """

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            # How many episodes of game environment to train network with.
            total_episodes = int(np.ceil(total_timesteps / self.env.timestep_limit))

            # Rate to update target network toward primary network.
            tau = self.train_freq / self.target_network_update_freq

            target_ops = updateTargetGraph(self.params, tau)

            # Create old episodes buffer.
            buffer_size_eps = int(np.ceil(self.buffer_size * self.n_agents / self.env.timestep_limit))
            old_episodes_buffer = ReplayBuffer(buffer_size=buffer_size_eps)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]

            with self.sess as sess:
                for n_ep in range(total_episodes):
                    current_episode_buffer = []
                    state = []
                    for na in range(self.n_agents):
                        current_episode_buffer.append(ReplayBuffer(buffer_size=self.env.timestep_limit))
                        state.append((np.zeros([1, self.h_size]), np.zeros([1, self.h_size])))
                    # Reset environment and get first new observation.
                    obs = self.env.reset()
                    done = False
                    reset = True
                    ep_timestep = 0
                    # The Q-Network
                    while ep_timestep < self.env.timestep_limit:
                        if callback is not None:
                            # Only stop training if return value is False, not when it is None. This is for backwards
                            # compatibility with callbacks that have no return statement.
                            if callback(locals(), globals()) is False:
                                break
                        ep_timestep += 1
                        epsilon = self.exploration.value(self.num_timesteps)

                        a = np.zeros(self.n_agents)
                        for na in range(self.n_agents):
                            # Choose an action greedily (with epsilon chance of random action) from the Q-network.
                            if np.random.rand(1) < epsilon or self.num_timesteps < self.learning_starts:
                                state[na] = sess.run(self.main_qnet.lstm_state,
                                                     feed_dict={self.main_qnet.scalar_input: [obs[na]],
                                                                self.main_qnet.lstm_input_size: 1,
                                                                self.main_qnet.lstm_state_in: state[na],
                                                                self.main_qnet.batch_size: 1})
                                a[na] = np.random.randint(0, self.env.action_space.n)
                            else:
                                a[na], state[na] = sess.run([self.main_qnet.predict, self.main_qnet.lstm_state],
                                                            feed_dict={self.main_qnet.scalar_input: [obs[na]],
                                                                       self.main_qnet.lstm_input_size: 1,
                                                                       self.main_qnet.lstm_state_in: state[na],
                                                                       self.main_qnet.batch_size: 1})

                        self.num_timesteps += 1

                        # Save the experience to our episode buffer.
                        new_obs, rew, done, _ = self.env.step(a)
                        reset = False
                        for na in range(self.n_agents):
                            experience = np.expand_dims(np.array([obs[na], a[na], rew[na], new_obs[na], done]), 0)
                            current_episode_buffer[na].add(experience)

                        if self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq == 0:
                            # Double-DQN
                            initial_state = (np.zeros([self.batch_size, self.h_size]),
                                             np.zeros([self.batch_size, self.h_size]))

                            batch = old_episodes_buffer.sample(self.batch_size, self.trace_length)
                            stacked_obs = np.vstack(batch[:, 3])

                            action_ = sess.run(self.main_qnet.predict,
                                               feed_dict={self.main_qnet.scalar_input: stacked_obs,
                                                          self.main_qnet.lstm_input_size: self.trace_length,
                                                          self.main_qnet.lstm_state_in: initial_state,
                                                          self.main_qnet.batch_size: self.batch_size})
                            q_ = sess.run(self.target_qnet.q_out,
                                          feed_dict={self.target_qnet.scalar_input: stacked_obs,
                                                     self.target_qnet.lstm_input_size: self.trace_length,
                                                     self.target_qnet.lstm_state_in: initial_state,
                                                     self.target_qnet.batch_size: self.batch_size})
                            next_q = q_[range(self.batch_size * self.trace_length), action_]
                            done_mask = 1 - batch[:, 4]
                            target_q = batch[:, 2] + done_mask * self.gamma * next_q

                            sess.run(self.main_qnet.updateModel,
                                     feed_dict={self.main_qnet.scalar_input: np.vstack(batch[:, 0]),
                                                self.main_qnet.target_q: target_q,
                                                self.main_qnet.actions: batch[:, 1],
                                                self.main_qnet.lstm_input_size: self.trace_length,
                                                self.main_qnet.lstm_state_in: initial_state,
                                                self.main_qnet.batch_size: self.batch_size})

                            updateTarget(target_ops, sess)  # Update the target network toward the primary network.

                        obs = new_obs
                        for i in range(self.n_agents):
                            episode_rewards[-1] += rew[i] # Sum of rewards for logging.
                        if done:
                            episode_rewards.append(0.0)
                            num_episodes = len(episode_rewards)
                            break

                    # Add the episode to the experience buffer
                    for na in range(self.n_agents):
                        old_episodes_buffer.add(current_episode_buffer[na].buffer)

    def predict(self, observations, states=None, mask=None, deterministic=False):
        """
        Predict function
        """

        with self.sess.as_default():
            actions = np.zeros(self.n_agents)
            if states is None:
                states = []
                for na in range(self.n_agents):
                    states.append((np.zeros([1, self.h_size]), np.zeros([1, self.h_size])))

            for na in range(self.n_agents):
                actions[na], states[na] = self.sess.run([self.main_qnet.predict, self.main_qnet.lstm_state],
                                                  feed_dict={self.main_qnet.scalar_input: [observations[na]],
                                                             self.main_qnet.lstm_input_size: 1,
                                                             self.main_qnet.lstm_state_in: states[na],
                                                             self.main_qnet.batch_size: 1})
        return actions, states

    def action_probability(self, observation, state=None, mask=None, actions=None):
        pass

    def save(self, save_path):
        """
        Save function
        """

        data = {
            "checkpoint_path": self.checkpoint_path,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "n_agents": self.n_agents,
            "magent": self.magent,
            "trace_length": self.trace_length,
            "h_size": self.h_size
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load function.
        """

        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
