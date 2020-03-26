import numpy as np
import random

class Parameters():
    '''
    - input a dict
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params 
    
    def __getattr__(self, x):
        return self.params[x.lower()]


class Trajectories():
    """
    only for one actor case
    """
    def __init__(self, len_=None):

        self.len_ = len_
        self.unrolled = None

        # Store when interact with env
        self.s = None
        self.a = None
        self.r = None
        self.log_a = None
        self.not_done = None
        self.ture_avg_return=None

        # cul later
        self.old_v = None
        self.adv = None
        

def discount(array, ecoff):
    ans = np.zeros_like(array)
    acc = 0.0
    for i in range(len(ans)-1, -1, -1):
        ans[i] = array[i] + acc * ecoff # shape (1,)
        acc = ans[i]
    return ans
    
# logs\03-26_01-43_maddpg_\model
class ReplayBuffer:
    """
    """
    def __init__(self, size):
        self.max_size = size
        self.buffer = []

    def size(self):
        return len(self.buffer)

    def store(self, step):
        self.buffer.append(step)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, num): 
        if num >= len(self.buffer):
            return self.buffer
        return random.sample(self.buffer, num)

    def gen_index(self, num):
        return np.random.randint(0, self.size(), num)

    def sample_index(self, index):
        s, a, r, s_next, done = [], [], [], [], []
        for idx in index:
            sample = self.buffer[idx]
            s.append(sample[0])
            a.append(sample[1])
            r.append([sample[2]])
            s_next.append(sample[3])
            done.append([sample[4]])
        return np.array(s), np.array(a), np.array(r), \
               np.array(s_next), np.array(done)