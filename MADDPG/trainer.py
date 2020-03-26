
from utils import Parameters
import time
import numpy as np
import tensorflow as tf

from ma_env import make_env
from agent import Agent

# todo
Test = False
best_ret = -1000.0
time_str = time.strftime("%m-%d_%H-%M", time.localtime())
info = ""
log_dir = "logs/" + "_".join([time_str, "maddpg", info])
if not Test:
    fw = tf.summary.create_file_writer(log_dir)
summary_step = 0

class Trainer():

    def __init__(self, params: Parameters):

        self.parms = params

        # Env
        self.env = make_env(params.game, None)
        obs_shapes = [self.env.observation_space[i].shape for i in range(self.env.n)]
        a_shapes   = [(self.env.action_space[i].n, ) for i in range(self.env.n)]
        a_bound    = params.a_bound
 
        # Seed
        self.env.seed(params.seed)
        np.random.seed(params.seed)
        tf.random.set_seed(params.seed)

        # Agents
        self.agents = [
            Agent(
                params=params,
                obs_shapes=obs_shapes,
                a_shapes=a_shapes,
                a_bounds=a_bound,
                a_shape_index=i
            ) for i in range(params.n_agent)
        ]


    def train(self):

        # Episode infomation
        episode_ret = [0.0]
        agent_epside_ret = [[0.0] for _ in range(self.parms.n_agent)]

        # Initialize s
        train_step_num = 0
        episode_num = 0
        episode_step = 0
        obs_n = self.env.reset()

        # todo
        if Test:
            self.agents[0].load_model("logs/03-26_01-43_maddpg_/model")
            self.agents[1].load_model("logs/03-26_01-43_maddpg_/model")

        while episode_num < self.parms.train_ep:
            # Interact
            pi_n, a_n = [], []
            for i in range(self.parms.n_agent):
                pi_n.append(self.agents[i].actor(np.array([obs_n[i]])))
                a_n.append(pi_n[i].sample()[0])
            # todo
            if Test:
                self.env.render()
                time.sleep(0.1)

            obs_next_n, r_n, done_n, _ = self.env.step(a_n)

            # Store
            for i, agent in enumerate(self.agents):
                agent.buffer.store(
                    (obs_n[i], a_n[i], r_n[i], obs_next_n[i], float(done_n[i])))

            # Train and record rewards        
            if agent.buffer.size() > self.parms.start_size:
                for i, agent in enumerate(self.agents):
                    agent._train(self.agents) # TODO
                    
                train_step_num += 1

            for i in range(self.parms.n_agent):
                episode_ret[-1] += r_n[i]
                agent_epside_ret[i][-1] += r_n[i]
                
                
            terminal = (episode_step >= self.parms.train_ep_len)

            if all(done_n) or terminal:
                # TODO 
                episode_num += 1
                print("epside", episode_num, 'end')
                obs_next_n = self.env.reset()
                episode_step = 0
                episode_ret.append(0)
                for a in agent_epside_ret:
                    a.append(0)

            obs_n = obs_next_n
            episode_step += 1

            if terminal and (episode_num + 1) % self.parms.check_rate == 0:
                # print
                rr = np.mean(episode_ret[-self.parms.check_rate:])
                print("episodes: {}, avg_ret: {}".format(
                    episode_num, 
                    rr
                ))
                global summary_step
                with fw.as_default():
                    tf.summary.scalar("return", rr, summary_step)
                    summary_step += 1
                # Save model
                global best_ret, log_dir
                if rr > best_ret:
                    best_ret = rr
                    for a in self.agents:
                        a.save_model(log_dir + "/model")
                

        return np.mean(episode_ret)
