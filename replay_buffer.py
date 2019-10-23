import random, pickle
import numpy as np

from queue import Queue
from collections import deque

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        self.per_e = 0.01
        self.per_a = 0.6
        self.per_b = 0.4
        self.per_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.
        self.max_priority = 1
        self.sum_priority = 0
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer_q = Queue()
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, *args):
        experience = []
        for arg in args:
            experience.append(arg)
        experience.append(self.max_priority)
        self.buffer_q.put(experience)

    def size(self):
        n_queued = self.buffer_q.qsize()
        return self.count + n_queued

    def sample_batch(self, batch_size, rnd=True):

        # copy queued experience to buffer before sampling

        while not self.buffer_q.empty():
            experience = self.buffer_q.get()
            if self.count < self.buffer_size:
                self.count += 1
            else:
                self.buffer.popleft()
            self.buffer.append(experience)

        # sample a batch of experience

        batch = []
        n = np.minimum(self.count, batch_size)
        if n > 0:
            last_idx = len(self.buffer[0]) - 1
            priorities = np.array([_[last_idx] for _ in self.buffer])
            sampling_probabilities = priorities / np.sum(priorities)
            if rnd:
                idx_batch = np.random.choice(self.count, n, replace=False, p=sampling_probabilities)
            else:
                idx_batch = np.arange(n)
            self.per_b = np.min([1., self.per_b + self.per_b_increment_per_sampling])
            p_min = np.min(priorities) / np.sum(priorities)
            max_weight = (p_min * n) ** (-self.per_b)
            w_batch = np.empty((n, 1), dtype=np.float32)
            w_batch[:, 0] = np.power(n * sampling_probabilities[idx_batch], -self.per_b) / max_weight
            for j in range(last_idx):
                batch.append(np.array([self.buffer[i][j] for i in idx_batch]))
            batch.append(idx_batch)
            batch.append(w_batch)
        return batch

    def update_priorities(self, idx, abs_errors):
        abs_errors += self.per_e
        clipped_error = np.minimum(abs_errors, self.absolute_error_upper)
        priorities = np.power(clipped_error, self.per_a)
        last_idx = len(self.buffer[0]) - 1
        for i, j in enumerate(idx):
            self.buffer[j][last_idx] = priorities[i]

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def save_buffer(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, fname):
        with open(fname, 'rb') as f:
            buffer = pickle.load(f)
        if len(buffer) > self.buffer_size:
            b_start = len(buffer) - self.buffer_size
            self.count = self.buffer_size
        else:
            b_start = 0
            self.count = len(buffer)
        for i in range(0, self.count):
            self.buffer.append(buffer[b_start + i])