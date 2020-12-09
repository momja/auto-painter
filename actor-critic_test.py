import math
from os import environ
import gym
import gym_painting

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tf_agents.agents.random import random_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.environments.utils import validate_py_environment
from collections import OrderedDict
import tqdm

# eps = np.finfo(np.float32).eps.item()
# huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


# class ActorCritic(tf.keras.Model):
#     """ Combined actor-critic network"""

#     def __init__(
#         self,
#         num_actions,
#         num_hidden_units):
#         """Initialize"""
#         super().__init__()
#         self.common = layers.Dense(num_hidden_units, activation="relu")
#         self.actor = layers.Dense(num_actions)
#         self.critic = layers.Dense(1)
    
#     def call(self, inputs: tf.Tensor):
#         x = self.common(inputs)
#         return self.actor(x), self.critic(x)

# # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# # This would allow it to be included in a callable TensorFlow graph.

# def env_step(action: np.ndarray):
#   """Returns state, reward and done flag given an action."""

#   state, reward, done, _ = env.step(action)
#   return (state.astype(np.float32), 
#           np.array(reward, np.int32), 
#           np.array(done, np.int32))


# def tf_env_step(action: tf.Tensor):
#   return tf.numpy_function(env_step, [action], 
#                            [tf.float32, tf.int32, tf.int32])


# def run_episode(
#     initial_state: tf.Tensor,  
#     model: tf.keras.Model, 
#     max_steps: int):
#   """Runs a single episode to collect training data."""

#   action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#   values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#   rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

#   initial_state_shape = initial_state.shape
#   state = initial_state

#   for t in tf.range(max_steps):
#     # Convert state into a batched tensor (batch size = 1)
#     state = tf.expand_dims(state, 0)
  
#     # Run the model and to get action probabilities and critic value
#     action, value = model(state)
  
#     # Sample next action from the action probability distribution
#     # action = tf.random.categorical(action_logits_t, 1)[0,0]
#     # action_probs_t = tf.nn.softmax(action_logits_t)

#     # Store critic values
#     values = values.write(t, tf.squeeze(value))

#     # Store log probability of the action chosen
#     # action_probs = action_probs.write(t, action_probs_t[0, action])
  
#     # Apply action to the environment to get next state and reward
#     state, reward, done = tf_env_step(action)
#     state.set_shape(initial_state_shape)
  
#     # Store reward
#     rewards = rewards.write(t, reward)

#     if tf.cast(done, tf.bool):
#       break

#   action_probs = action_probs.stack()
#   values = values.stack()
#   rewards = rewards.stack()
  
#   return action_probs, values, rewards


# def compute_loss(
#     action_probs: tf.Tensor,  
#     values: tf.Tensor,  
#     returns: tf.Tensor) -> tf.Tensor:
#   """Computes the combined actor-critic loss."""

#   advantage = returns - values

#   action_log_probs = tf.math.log(action_probs)
#   actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

#   critic_loss = huber_loss(values, returns)

#   return actor_loss + critic_loss

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# @tf.function
# def train_step(
#     initial_state: tf.Tensor, 
#     model: tf.keras.Model, 
#     optimizer: tf.keras.optimizers.Optimizer, 
#     gamma: float, 
#     max_steps_per_episode: int) -> tf.Tensor:
#   """Runs a model training step."""

#   with tf.GradientTape() as tape:

#     # Run the model for one episode to collect training data
#     action_probs, values, rewards = run_episode(
#         initial_state, model, max_steps_per_episode) 

#     # Calculate expected returns
#     returns = get_expected_return(rewards, gamma)

#     # Convert training data to appropriate TF tensor shapes
#     action_probs, values, returns = [
#         tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

#     # Calculating loss values to update our network
#     loss = compute_loss(action_probs, values, returns)

#   # Compute the gradients from the loss
#   grads = tape.gradient(loss, model.trainable_variables)

#   # Apply the gradients to the model's parameters
#   optimizer.apply_gradients(zip(grads, model.trainable_variables))

#   episode_reward = tf.math.reduce_sum(rewards)

#   return episode_reward


# def get_expected_return(
#     rewards: tf.Tensor, 
#     gamma: float, 
#     standardize: bool = True) -> tf.Tensor:
#   """Compute expected returns per timestep."""

#   n = tf.shape(rewards)[0]
#   returns = tf.TensorArray(dtype=tf.float32, size=n)

#   # Start from the end of `rewards` and accumulate reward sums
#   # into the `returns` array
#   rewards = tf.cast(rewards[::-1], dtype=tf.float32)
#   discounted_sum = tf.constant(0.0)
#   discounted_sum_shape = discounted_sum.shape
#   for i in tf.range(n):
#     reward = rewards[i]
#     discounted_sum = reward + gamma * discounted_sum
#     discounted_sum.set_shape(discounted_sum_shape)
#     returns = returns.write(i, discounted_sum)
#   returns = returns.stack()[::-1]

#   if standardize:
#     returns = ((returns - tf.math.reduce_mean(returns)) / 
#                (tf.math.reduce_std(returns) + eps))

#   return returns

















if __name__ == "__main__":
    # train_env_tf = suite_gym.load("Painter-v0")
    # val_env_tf = suite_gym.load("Painter-v0")
    # train_env = tf_py_environment.TFPyEnvironment(train_env_tf)
    # val_env = tf_py_environment.TFPyEnvironment(val_env_tf)

    env = gym.make("Painter-v0")

    # validate_py_environment(train_env_tf)
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

    num_actions = gym.spaces.flatdim(env.action_space)
    num_hidden_units = 128
    model = ActorCritic(num_actions, num_hidden_units)

    max_episodes = 10000
    max_steps_per_episode = 5000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100 
    # consecutive trials
    reward_threshold = 1
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99
    i = 0
    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = int(train_step(
                initial_state, model, optimizer, gamma, max_steps_per_episode))

            running_reward = episode_reward*0.01 + running_reward*.99
        
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)
        
            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass # print(f'Episode {i}: average reward: {avg_reward}')
        
            if running_reward > reward_threshold:  
                break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
    
    # random_policy = random_tf_policy.RandomTFPolicy(
    #     train_env.time_step_spec(), train_env.action_spec()
    # )
    # time_step = train_env.reset()

    # while not time_step.is_last():
    #     action_step = random_policy.action(time_step)
    #     time_step = train_env.step(action_step.action)
    #     train_env_tf.render()
