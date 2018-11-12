from collections import deque
import random


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.memory = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        self.memory.append(transition)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.memory, batchsize)
        return samples

    def __len__(self):
        return len(self.memory)
