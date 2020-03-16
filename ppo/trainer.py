
from utils import Trajectories, Parameters, discount
from env import Env
from models import CtsPolicy, ValueNet
import numpy as np
import tensorflow as tf

class Trainer():
    """
    Initilaize an agent based on params,
    collect steps,
    train the agent according to the steps
    """

    def __init__(self, params: Parameters):
        self.parms = params

        self.env = Env(params.game, params.gamma)

        # Seed
        self.env.seed(params.seed)
        np.random.seed(params.seed)
        tf.random.set_seed(params.seed)

        self.critic = ValueNet(lr=params.lr_c)

        self.actor = CtsPolicy(action_bound=self.env.action_bound, 
                               action_dim=self.env.num_actions, lr=params.lr_a)

        tf.summary.trace_on(graph=True)
    
    def _collect_step(self, traj: Trajectories):
        """
        Complete (s, a, r, log_a) in traj

        Inputs:
        - Traj:  class
        """
        # Time step
        traj_len = traj.len_

        # Space for a, r, log_a, not_done
        shape = (traj_len, 1)
        all_zeros = [np.zeros(shape) for _ in range(3)]
        a_buffer, log_a_buffer, r_buffer = all_zeros
        not_done_buffer = np.ones(shape, dtype=int)

        # Space for s
        state_shape = (traj_len + 1, ) + self.env.features_shape
        s_buffer = np.zeros(state_shape)

        # Episode infomation
        episode_ret = []

        # Initialize s
        s = self.env.reset()
        for t in range(traj_len):
            # Interact
            pi = self.actor(s[np.newaxis, :]) # batch_size=1
            a = pi.sample()[0] # shape (1, )
            s_next, r, done, info = self.env.step(a)

            # Store
            s_buffer[t, :] = s
            a_buffer[t] = a
            r_buffer[t] = [r]
            log_a_buffer[t] = pi.log_prob(a)[0].numpy()
                        
            if done:
                not_done_buffer[t] = [0] # V(next_s) = 0
                _, ret = info['done']
                episode_ret.append(ret)
                s_next = self.env.reset() # don't need to store terminal state

            s = s_next

        # Store the last state in order to cal V(s_next)
        s_buffer[traj_len] = s
        
        # Store into traj
        traj.s = s_buffer
        traj.a = a_buffer
        traj.r = r_buffer
        traj.log_a = log_a_buffer
        traj.not_done = not_done_buffer

        traj.ture_avg_return = np.mean(episode_ret)


    def _cal_adv_and_old_v(self, traj: Trajectories):
        """
        Complete adv and old_v in traj
        """
        v = self.critic(traj.s)
        v_s = v[:-1,:]
        v_s_next = v[1:,:]
        td = traj.r - v_s + self.parms.gamma * v_s_next * traj.not_done
        
        advs = np.zeros_like(td)

        # Find done index
        done_index = np.where(traj.not_done == 0)[0]
        
        # Cal adv for each intervel
        start_index = 0
        multiplier = self.parms.gamma * self.parms.lamda
        for end_index in done_index:
            advs[start_index:end_index+1,...] = discount(
                td[start_index:end_index+1,...], multiplier)
            start_index = end_index + 1

        traj.adv = advs
        traj.old_v = v_s.numpy()


    def _train_critic(self, traj: Trajectories):
        # Split traj into batches
        state_indices = np.arange(traj.len_)
        np.random.shuffle(state_indices)
        n_batch = traj.len_ // self.parms.batch_size
        batches_index = np.array_split(state_indices, n_batch)

        for batch_index in batches_index:
            for _ in range(self.parms.critic_epoch):
                # with tf.name_scope('test') as scope:
                with tf.GradientTape() as tape:
                    # Get new_v
                    new_v = self.critic(traj.s[batch_index])
                    # Get loss: old_v + adv - new_v
                    loss = traj.adv[batch_index] + traj.old_v[batch_index] - new_v
                    
                    loss = tf.reduce_mean(tf.square(loss))
                grads = tape.gradient(loss, self.critic.trainable_weights)
                grads = [tf.clip_by_norm(grad, 10.0) for grad in grads]
                self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

    
    def _train_actor(self, traj: Trajectories):

        # Split traj into batches
        state_indices = np.arange(traj.len_)
        np.random.shuffle(state_indices)
        n_batch = traj.len_ // self.parms.batch_size
        batches_index = np.array_split(state_indices, n_batch)
        
        for batch_index in batches_index:
            for _ in range(self.parms.actor_epoch):
                with tf.GradientTape() as tape:
                    # Get ratio
                    new_pi = self.actor(traj.s[batch_index])
                    new_log_a = new_pi.log_prob(traj.a[batch_index])
                    ratio = tf.exp(new_log_a - traj.log_a[batch_index])
                    ratio = tf.clip_by_value(ratio, 1.0 - self.parms.clip_ep, 
                                            1.0 + self.parms.clip_ep)
                    # Get surr
                    surr = traj.adv[batch_index] * ratio
                    entropy = self.parms.entropy_ecoff * new_pi.entropy()
                    # Get loss
                    loss = - (surr + entropy)
                grads = tape.gradient(loss, self.actor.trainable_weights)
                grads = [tf.clip_by_norm(grad, 10.0) for grad in grads]
                self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

    def train_step(self):

        traj = Trajectories(self.parms.train_step_len)

        # with tf.name_scope('collect_step') as scope:
        self._collect_step(traj)

        # with tf.name_scope('cal_adv_and_old_v') as scope:
        self._cal_adv_and_old_v(traj)

        # with tf.name_scope('train_critic') as scope:
        self._train_critic(traj)

        # with tf.name_scope('train_actor') as scope:
        self._train_actor(traj)

        return traj.ture_avg_return
        
