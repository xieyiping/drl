import numpy as np
import time

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env






if __name__ == "__main__":
    """
    test make_env func
    """

    env = make_env("simple_speaker_listener", None)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    act_shape_n = [(env.action_space[i].n,) for i in range(env.n)]
    env.reset()

    for _ in range(1000):
        
        env.render()
        time.sleep(0.1)
        action_sample = [np.random.rand(*a_shape) for a_shape in act_shape_n]
        s, r, d, info = env.step(action_sample)
        if all(d):
            env.reset()
    # print(action_sample)