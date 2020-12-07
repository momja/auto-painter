import math
from os import environ
import gym
import gym_painting

import tensorflow as tf
import numpy as np
from tf_agents.agents.random import random_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from collections import OrderedDict

if __name__ == "__main__":
    environment_tf = suite_gym.load("Painter-v0")
    environment = tf_py_environment.TFPyEnvironment(environment_tf)
    # environment = tf_py_environment.TFPyEnvironment(environment)
    # print('action_spec:', environment.action_spec())
    # print('time_step_spec.observation:', environment.time_step_spec().observation)
    # print('time_step_spec.step_type:', environment.time_step_spec().step_type)
    # print('time_step_spec.discount:', environment.time_step_spec().discount)
    # print('time_step_spec.reward:', environment.time_step_spec().reward)


    # action = OrderedDict(
    #     [
    #         ("color", np.array([1, 1, 1])),
    #         ("motion", np.array([math.pi / 4, 4, 5])),
    #         ("pendown", 1),
    #     ]
    # )
    random_policy = random_tf_policy.RandomTFPolicy(
        environment.time_step_spec(), environment.action_spec()
    )
    time_step = environment.reset()

    while not time_step.is_last():
        action_step = random_policy.action(time_step)
        time_step = environment.step(action_step.action)
        environment_tf.render()
