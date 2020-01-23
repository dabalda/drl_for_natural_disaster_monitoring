import numpy as np

class ReplayBuffer(object):
    """
    Replay buffer class for the DRQN algorithm.

    :param buffer_size: (int) the maximum amount of experience samples that the buffer can hold.
    """

    def __init__(self, buffer_size=100):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        """
        Add function.

        :param experience: (any) experience sample to add to the buffer.
        """

        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        """
        Sample function.

        :param batch_size: (int) how many traces to sample.
        :param trace_length: (int) the size of each trace of consecutive experience samples.
        """
        sampled_episodes_idx = np.random.randint(low=0, high=len(self.buffer)-1, size=batch_size)
        sampled_traces = []
        for ep_idx in sampled_episodes_idx:
            len_ep = len(self.buffer[ep_idx])
            point = np.random.randint(low=0, high=len_ep + 1 - trace_length)
            sampled_traces.append(self.buffer[ep_idx][point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 5])  # Will be reshaped back in the network
