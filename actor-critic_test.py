import gym
import gym_painting 

import tensorflow as tf
import numpy as np
from tf_agents.agents.random import random_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

if __name__ == "__main__":
    environment = suite_gym.load('Painter-v0')
    # environment = tf_py_environment.TFPyEnvironment(environment)
    # print('action_spec:', environment.action_spec())
    # print('time_step_spec.observation:', environment.time_step_spec().observation)
    # print('time_step_spec.step_type:', environment.time_step_spec().step_type)
    # print('time_step_spec.discount:', environment.time_step_spec().discount)
    # print('time_step_spec.reward:', environment.time_step_spec().reward)

    from collections import OrderedDict
    action = OrderedDict([('color', np.array([1,1,1])),('motion', np.array([0,1,5])),('pendown', 1)])
    # action = np.array([0.1])
    time_step = environment.reset()
    while not time_step.is_last():
        time_step = environment.step(action)
        environment.render()
        print(time_step)
