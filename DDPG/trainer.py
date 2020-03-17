
from utils import ReplayBuffer, Parameters
from env import Env
from models import CtsPolicy, DDPGValueNet
import numpy as np
import tensorflow as tf

class Trainer():

    def __init__(self, params: Parameters):
        self.parms = params

        self.env = Env(params.game, params.gamma,
                       norm_rewards=None, norm_states=False)

        self.buffer = ReplayBuffer(params.replay_size)

        # Seed
        self.env.seed(params.seed)
        np.random.seed(params.seed)
        tf.random.set_seed(params.seed)

        self.critic = DDPGValueNet(feature_shape=self.env.features_shape,
                                   a_num=self.env.num_actions,lr=params.lr_c)
        self.target_critic = DDPGValueNet(feature_shape=self.env.features_shape,
                                   a_num=self.env.num_actions,lr=params.lr_c)
        self._copy_para(self.critic.model, self.target_critic.model)

        self.actor = CtsPolicy(action_bound=self.env.action_bound, 
                               action_dim=self.env.num_actions, lr=params.lr_a)
        self.target_actor = CtsPolicy(action_bound=self.env.action_bound, 
                               action_dim=self.env.num_actions, lr=params.lr_a)
        self._copy_para(self.actor, self.target_actor)

        self.ema = tf.train.ExponentialMovingAverage(decay=1.0 - self.parms.tau)


    def _copy_para(self, from_model, to_model):
        """
        Copy parameters for soft updating
        :param from_model: latest model
        :param to_model: target model
        :return: None
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)


    def _ema_update(self):

        paras = self.actor.trainable_weights + \
                self.critic.model.trainable_weights

        self.ema.apply(paras)

        for i, j in zip(self.target_actor.trainable_weights + \
            self.target_critic.model.trainable_weights, paras):
            i.assign(self.ema.average(j))
        

    def _train(self):

        # Sample
        batch = self.buffer.sample(self.parms.batch_size)
        s      = np.array([batch_[0] for batch_ in batch])        
        a      = np.array([batch_[1] for batch_ in batch])        
        r      = np.array([batch_[2] for batch_ in batch])        
        s_next = np.array([batch_[3] for batch_ in batch])        
        not_done = np.array([not batch_[4] for batch_ in batch])        

        # Reshpe
        r = r[:, np.newaxis]
        not_done = not_done[:, np.newaxis]

        # Train critic
        with tf.GradientTape() as tape:
            pi_next = self.target_actor(s_next)
            a_next = pi_next.sample()
            q_next = self.target_critic([s_next, a_next])
            y = r + self.parms.gamma * q_next * not_done
            q = self.critic([s, a])
            c_loss = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(c_loss, self.critic.model.trainable_weights)
        self.critic.model.optimizer.apply_gradients(zip(
            c_grads, self.critic.model.trainable_weights))

        # Train actor
        with tf.GradientTape() as tape:
            pi = self.actor(s)
            a = pi.sample()
            q = self.critic([s, a])
            a_loss = -tf.reduce_mean(q)
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(
            a_grads, self.actor.trainable_weights))
        
        self._ema_update()


    def train_step(self):

        # Episode infomation
        episode_ret = []

        # Initialize s
        s = self.env.reset()
        for _ in range(self.parms.train_step_len):
            # Interact
            pi = self.actor(s[np.newaxis, :]) # batch_size=1
            a = pi.sample()[0]
            s_next, r, done, info = self.env.step(a)

            # Store
            self.buffer.store((s, a, r, s_next, done))

            # Train
            if self.buffer.size() > self.parms.start_size:
                self._train()

            if done:
                _, ret = info['done']
                episode_ret.append(ret)
                s_next = self.env.reset()

            s = s_next
        
        return np.mean(episode_ret)