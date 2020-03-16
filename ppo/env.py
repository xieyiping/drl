import gym
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous


class RunningStat(): 
    """
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape) 
        self._S = np.zeros(shape) 
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape


class Identity():
    """
    A convenience class which simply implements __call__
    as the identity function
    """
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


class RewardFilter():
    """
    Incorrect reward normalization [copied from OAI code]
    update return
    divide reward by std(return) without subtracting and adding back mean
    optionally clip reward
    """
    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()


class ZFilter():
    """
    y = (x-mean)/std
    using running estimates of mean,std
    here to norm *state*
    """
    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)  
        self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs) 
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()


class Env():
    """
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    """
    def __init__(self, game, gamma=0.95, norm_states=True, norm_rewards='returns',
                 clip_obs=-1, clip_rew=-1):
        self.env = gym.make(game)
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
                            
        self.action_bound = self.env.action_space.high
        
        # Number of features
        self.features_shape = self.env.reset().shape

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=self.features_shape, \
                                            clip=clip_obs)

        # Support for rewards normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), 
                                         center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), 
                                              gamma=gamma, clip=clip_rew)


    def reset(self):
        # Reset the state, and the running total reward
        start_state = self.env.reset()
        start_state = self.state_filter(start_state)
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.state_filter.reset()
        self.reward_filter.reset()
        return start_state


    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        self.counter += 1
        state = self.state_filter(state)
        _reward = self.reward_filter(reward)
        self.total_true_reward += reward
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info


    def seed(self, seed_):
        self.env.seed(seed_)