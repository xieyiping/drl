
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
        
        # Four critic nets
        critic_nets = [DDPGValueNet(feature_shape=self.env.features_shape,
                       a_num=self.env.num_actions,lr=params.lr_c) for _ in range(4)]
        self.critic1, self.critic2, self.target_critic1, self.target_critic2 = critic_nets   
                                
        # Two actor nets
        self.actor = CtsPolicy(action_bound=self.env.action_bound, 
                               action_dim=self.env.num_actions, lr=params.lr_a)
        self.target_actor = CtsPolicy(action_bound=self.env.action_bound, 
                               action_dim=self.env.num_actions, lr=params.lr_a)

        # Copy parms
        self._copy_para(self.critic1, self.target_critic1)
        self._copy_para(self.critic2, self.target_critic2)
        self._copy_para(self.actor, self.target_actor)

        self.train_step_cnt = 0


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

        # Set target y
        pi_next = self.target_actor(s_next)
        a_next = pi_next.sample()
        q_next = tf.minimum(self.target_critic1([s_next, a_next]),
                            self.target_critic2([s_next, a_next]))
        y = r + self.parms.gamma * q_next * not_done

        # Train critic1
        with tf.GradientTape() as c1_tape:
            q1 = self.critic1([s, a])
            c1_loss = tf.losses.mean_squared_error(y, q1)
        c1_grads = c1_tape.gradient(c1_loss, self.critic1.trainable_weights)
        self.critic1.optimizer.apply_gradients(zip(
            c1_grads, self.critic1.trainable_weights))
        
        # Train critic2
        with tf.GradientTape() as c2_tape:
            q2 = self.critic2([s, a])
            c2_loss = tf.losses.mean_squared_error(y, q2)
        c2_grads = c2_tape.gradient(c2_loss, self.critic2.trainable_weights)
        self.critic2.optimizer.apply_gradients(zip(
            c2_grads, self.critic2.trainable_weights))

        # Train actor
        if self.train_step_cnt % self.parms.actor_interval == 0:

            with tf.GradientTape() as a_tape:
                pi = self.actor(s)
                a = pi.sample()
                q = self.critic1([s, a])
                a_loss = -tf.reduce_mean(q)
            a_grads = a_tape.gradient(a_loss, self.actor.trainable_weights)
            self.actor.optimizer.apply_gradients(zip(
                a_grads, self.actor.trainable_weights))

            # update parms
            self._target_soft_update(self.actor, self.target_actor)
            self._target_soft_update(self.critic1, self.target_critic1)
            self._target_soft_update(self.critic2, self.target_critic2)


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
                self.train_step_cnt += 1

            if done:
                _, ret = info['done']
                episode_ret.append(ret)
                s_next = self.env.reset()

            s = s_next
        
        return np.mean(episode_ret)