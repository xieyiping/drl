import numpy as np

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
    def __init__(
        self, 
        len_=None,
        states=None, # for  V,adv
        rewards=None, # for adv
        not_dones=None, # for td->adv
        actions=None, # for log(a)
        log_a=None, # for surr->actor
        advantages=None,
        unrolled=False, 
        values=None): # for acritic

        self.len_ = len_
        self.unrolled = unrolled

        # Store when interact with env
        self.s = states
        self.a = actions
        self.r = rewards
        self.log_a = log_a
        self.not_done = not_dones
        self.ture_avg_return=None

        # cul later
        self.old_v = values
        self.adv = advantages
        

def discount(array, ecoff):
    ans = np.zeros_like(array)
    acc = 0.0
    for i in range(len(ans)-1, -1, -1):
        ans[i] = array[i] + acc * ecoff # shape (1,)
        acc = ans[i]
    return ans
    