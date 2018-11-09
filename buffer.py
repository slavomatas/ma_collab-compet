import random
from collections import deque

from utils import transpose_list


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition):
        """push into the buffer"""

        self.deque.append(transition)

        '''
        input_to_buffer = transpose_list(transition)

        for item in input_to_buffer:
            self.deque.append(item)
        '''

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        # return transpose_list(samples)

        return samples

    def __len__(self):
        return len(self.deque)
