from collections import namedtuple
import random
import numpy as np

class ReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward','terminate_flag','time_steps', 'valid_action_num'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.default_rng().choice(len(self.memory), batch_size, replace=False)
        data = [self.memory[idx] for idx in indices]
        return data

    def __len__(self):
        return len(self.memory)



class Step_ReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward','terminate_flag','time_steps', 'valid_action_num'))
        self.capacity = capacity
        self.memory = [ReplayMemory(10000) for _ in range(24)]

    def push(self, *args):
        """Saves a transition."""

        state=self.Transition(*args)
        sub_memory=self.memory[state.state[0]//3600%24]
        if len(sub_memory.memory) < sub_memory.capacity:
            sub_memory.memory.append(None)
        sub_memory.memory[sub_memory.position] = state
        sub_memory.position = (sub_memory.position + 1) % sub_memory.capacity

    def sample(self, batch_size):
        bs=batch_size//24
        data=[]
        for m in self.memory:
            indices = np.random.default_rng().choice(len(m.memory), bs, replace=False)
            data += [m.memory[idx] for idx in indices]
        return data

    def __len__(self):
        return len(self.memory[0].memory)



class Step_PrimeReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state','trip_flag','time_steps', 'valid_action_num'))
        self.capacity = capacity
        self.memory = [Prime_ReplayMemory(15000) for _ in range(24)]

    def push(self, *args):
        """Saves a transition."""

        state=self.Transition(*args)
        sub_memory=self.memory[state.state[0]//3600%24]
        if len(sub_memory.memory) < sub_memory.capacity:
            sub_memory.memory.append(None)
        sub_memory.memory[sub_memory.position] = state
        sub_memory.position = (sub_memory.position + 1) % sub_memory.capacity

    def sample(self, batch_size):
        bs=batch_size//24
        data=[]
        for m in self.memory:
            indices = np.random.default_rng().choice(len(m.memory), bs, replace=False)
            data += [m.memory[idx] for idx in indices]
        return data

    def __len__(self):
        return len(self.memory[0].memory)


class Step_FReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'next_state','on_opt'))
        self.capacity = capacity
        self.memory = [F_ReplayMemory(15000) for _ in range(24)]

    def push(self, *args):
        """Saves a transition."""

        state=self.Transition(*args)
        sub_memory=self.memory[state.state[0]//3600%24]
        if len(sub_memory.memory) < sub_memory.capacity:
            sub_memory.memory.append(None)
        sub_memory.memory[sub_memory.position] = state
        sub_memory.position = (sub_memory.position + 1) % sub_memory.capacity

    def sample(self, batch_size):
        bs=batch_size//24
        data=[]
        for m in self.memory:
            indices = np.random.default_rng().choice(len(m.memory), bs, replace=False)
            data += [m.memory[idx] for idx in indices]
        return data

    def __len__(self):
        return len(self.memory[0].memory)



class Prime_ReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state','trip_flag','time_steps', 'valid_action_num'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.default_rng().choice(len(self.memory), batch_size, replace=False)
        data = [self.memory[idx] for idx in indices]
        return data
        #return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class F_ReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'next_state','on_opt'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size):
    #     # return self.memory
    #     return random.sample(self.memory, batch_size)

    def sample(self, batch_size):
        indices = np.random.default_rng().choice(len(self.memory), batch_size, replace=False)
        data = [self.memory[idx] for idx in indices]
        return data

    def reset(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

class Trajectory_ReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('trajectory'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.default_rng().choice(len(self.memory), batch_size, replace=False)
        data = [self.memory[idx] for idx in indices]
        return data
        #return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# class PrioritizedMemory:   # stored as ( s, a, r, s_ , t)
#     """
#     prioritized by error; see Schaul T. et al. - Prioritized experience replay, 2016
#     From https://github.com/jaromiru/ai_examples/blob/master/open_gym/utils.py with some modifications
#     n-step rewards; see Sutton, R. S., and Barto, A. G. 1998. Reinforcement Learning: An Introduction
#     """
#     e = 0.1
#     b = 0.6
#
#
#     def __init__(self, capacity, n_step = 4):
#         self.tree = SumTree(capacity)
#         self.capacity = capacity
#         self.len = 0
#         self.Transition = namedtuple('Transition',
#                                      ('state', 'action', 'next_state', 'reward'))
#         self.n_step = n_step
#
#         self.transition_buffer = []
#         self.error_buffer = []
#         self.cumulated_reward = 0
#
#     def __len__(self):
#         return self.len
#
#     def _add(self):
#         """ Add the oldest transition of the buffer to the memory """
#         t = self.transition_buffer[0]
#         last_t = self.transition_buffer[-1]
#         transition = self.Transition(t.state, t.action, self.cumulated_reward, last_t.next_state)
#         self.tree.add(self.error_buffer[0], transition)
#
#         self.cumulated_reward -= self.transition_buffer[0].reward
#         self.transition_buffer = self.transition_buffer[1:]
#         self.error_buffer = self.error_buffer[1:]
#
#     def add(self, error, *args):
#         """
#         Add the transition to the buffer.
#         Add the oldest state to the memory if the buffer of size $n_step is full.
#         Add the entire buffer to the memory if the state is terminal
#         """
#         self.len = min(self.capacity, self.len+1)
#
#         transition = self.Transition(*args)
#         self.transition_buffer.append(transition)
#         self.error_buffer.append(error)
#         self.cumulated_reward += transition.reward
#
#     def sample(self, n):
#         batch = []
#         segment = self.tree.total() / n
#
#         for i in range(n):
#             a = segment * i
#             b = segment * (i + 1)
#
#             s = np.random.uniform(a, b)
#             (idx, p, data) = self.tree.get(s)
#             batch.append( (idx, data) )
#
#         return batch
#
#     def update(self, idx, error):
#         self.tree.update(idx, self._compute_priority(error))
#
#     def _compute_priority(self, error):
#         return (error + self.e) ** self.b
#
# class SumTree:
#     """From https://github.com/jaromiru/ai_examples/blob/master/open_gym/utils.py"""
#     write = 0
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = np.zeros( 2*capacity - 1 )
#         self.data = np.zeros( capacity, dtype=object )
#
#     def _propagate(self, idx, change):
#         parent = (idx - 1) // 2
#
#         self.tree[parent] += change
#
#         if parent != 0:
#             self._propagate(parent, change)
#
#     def _retrieve(self, idx, s):
#         left = 2 * idx + 1
#         right = left + 1
#
#         if left >= len(self.tree):
#             return idx
#
#         if s <= self.tree[left]:
#             return self._retrieve(left, s)
#         else:
#             return self._retrieve(right, s-self.tree[left])
#
#     def total(self):
#         return self.tree[0]
#
#     def add(self, p, data):
#         idx = self.write + self.capacity - 1
#
#         self.data[self.write] = data
#         self.update(idx, p)
#
#         self.write += 1
#         if self.write >= self.capacity:
#             self.write = 0
#
#     def update(self, idx, p):
#         change = p - self.tree[idx]
#
#         self.tree[idx] = p
#         self._propagate(idx, change)
#
#     def get(self, s):
#         idx = self._retrieve(0, s)
#         dataIdx = idx - self.capacity + 1
#
#         return (idx, self.tree[idx], self.data[dataIdx])
