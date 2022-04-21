from constants import constants
from collections import deque
import random


class ReplayMemory:
    def __init__(self, capacity=constants['memory']['capacity']):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def sample(self, batchSize=constants['memory']['batchSize']):
        state, action, reward, nextState, done = zip(
            *random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = deque(maxlen=capacity)
