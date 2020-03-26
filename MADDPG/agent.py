from models import CtsPolicy, DDPGValueNet
from utils import ReplayBuffer, Parameters
import numpy as np
import tensorflow as tf
import os

class Agent():
    def __init__(self, 
                 params: Parameters, 
                 obs_shapes,
                 a_shapes,
                 a_bounds,
                 a_shape_index):
        self.index = a_shape_index
        self.parms = params

        self.buffer = ReplayBuffer(params.replay_size)
        
        # Critic
        self.critic = DDPGValueNet(feature_shapes=obs_shapes,
                                   a_shapes=a_shapes,lr=params.lr_c, 
                                   n_agent=params.n_agent)

        self.target_critic = DDPGValueNet(feature_shapes=obs_shapes,
                                   a_shapes=a_shapes,lr=params.lr_c, 
                                   n_agent=params.n_agent)

        self._copy_para(self.critic.model, self.target_critic.model)


        # Actor
        self.actor = CtsPolicy(action_bound=a_bounds,
                               s_shape=obs_shapes[a_shape_index],
                               a_shape=a_shapes[a_shape_index], 
                               lr=params.lr_a)

        self.target_actor = CtsPolicy(action_bound=a_bounds,
                               s_shape=obs_shapes[a_shape_index],
                               a_shape=a_shapes[a_shape_index], 
                               lr=params.lr_a)

        self._copy_para(self.actor, self.target_actor)


    def _copy_para(self, from_model, to_model):
        """
        Copy parameters for soft updating
        :param from_model: latest model
        :param to_model: target model
        :return: None
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)


    def _target_soft_update(self, net, target_net):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - self.parms.tau) + param * self.parms.tau
            )


    def _train(self, agents):
        # Index
        index = self.buffer.gen_index(self.parms.batch_size)

        # Sample
        obs_n = []
        obs_next_n = []
        act_n = []

        for a in agents:
            obs, act, _, obs_next, _ = a.buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)

        obs, act, r, obs_next, done = self.buffer.sample_index(index)

        # Train critic
        a_next_n = [a.target_actor(s).sample() for a, s in zip(agents, obs_n)]
        q_next = self.target_critic(obs_next_n + a_next_n)
        y = r + self.parms.gamma * q_next * (1.0 - done)
        with tf.GradientTape() as tape:
            q = self.critic(obs_n + act_n)
            c_loss = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(c_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(
            c_grads, self.critic.trainable_weights))

        # Train actor
        with tf.GradientTape() as tape:
            pi = self.actor(obs)
            a_n = pi.sample()
            act_n[self.index] = a_n
            q = self.critic(obs_n + act_n)
            a_loss = -tf.reduce_mean(q)
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(
            a_grads, self.actor.trainable_weights))
        
        self._target_soft_update(self.actor, self.target_actor)
        self._target_soft_update(self.critic, self.target_critic)


    def load_model(self, dir_="./models"):
        # if not os.path.exists(dir_):
        #     os.makedirs(dir_)
        self.actor.model.load_weights(
            dir_ + "/actor_model_" + str(self.index) + ".hdf5")
        self.critic.model.load_weights(
            dir_ + "/critic_model_" + str(self.index) + ".hdf5")

    def save_model(self, dir_="./models"):
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        self.actor.model.save_weights(
            dir_ + "/actor_model_" + str(self.index) + ".hdf5")
        self.critic.model.save_weights(
            dir_ + "/critic_model_" + str(self.index) + ".hdf5")
