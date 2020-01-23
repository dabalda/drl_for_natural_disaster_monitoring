from algorithms.ma_drqn.replay_buffer import ReplayBuffer
import numpy as np

old_episodes_buffer_size = 6
n_episodes = 10
n_agents = 2
ep_timestep_limit = 20
batch_size = 3
trace_length = 4

old_episodes_buffer = ReplayBuffer(buffer_size=old_episodes_buffer_size)

for n_ep in range(n_episodes):
    current_episode_buffer = []

    for na in range(n_agents):
        current_episode_buffer.append(ReplayBuffer(buffer_size=ep_timestep_limit))

    ep_timestep = 0
    while ep_timestep < ep_timestep_limit:
        ep_timestep += 1

        for na in range(n_agents):
            experience = np.random.rand(5,1)
            current_episode_buffer[na].add(experience)

        if n_ep >= 1:
            batch = old_episodes_buffer.sample(batch_size, trace_length)

    for na in range(n_agents):
        old_episodes_buffer.add(current_episode_buffer[na].buffer)