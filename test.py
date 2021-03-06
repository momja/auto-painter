import base64
import imageio
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import tempfile
import reverb
import PIL.Image

import gym
import gym_painting

import tensorflow as tf

from painting_critic_network import PaintingCriticNetwork
from painting_actor_network import PaintingActorNetwork

import tf_agents
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_gym
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils

from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network, nest_map, sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

tempdir = tempfile.gettempdir()

env_name = "Painter-v0"
collect_env = suite_gym.load(env_name)
# collect_env = TFPyEnvironment(collect_env)
eval_env = suite_gym.load(env_name)
# eval_env = TFPyEnvironment(eval_env)

# validate_py_environment(collect_env)

reverb_port = 8080

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 10000  # @param {type:"integer"}

# initial_collect_steps = 10000  # @param {type:"integer"}
initial_collect_steps = 100
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 10000  # @param {type:"integer"}

batch_size = 1  # @param {type:"integer"}

critic_learning_rate = 1e-3  # @param {type:"number"}
actor_learning_rate = 1e-3  # @param {type:"number"}
alpha_learning_rate = 1e-3
target_update_tau = 0.005  # @param {type:"number"}
target_update_period = 1  # @param {type:"number"}
gamma = 0.99  # @param {type:"number"}
reward_scale_factor = 1.0  # @param {type:"number"}

actor_fc_layer_params = (256, 256)
# critic_joint_fc_layer_params = (256, 256)

log_interval = 1000  # @param {type:"integer"}

num_eval_episodes = 20  # @param {type:"integer"}
eval_interval = 10000  # @param {type:"integer"}

policy_save_interval = 5000

use_gpu = True
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)

# observation space:
# - patch: (n,m,3) where n and m are observation shape
# - color: (1,3)
# - motion: (1,2)
# - pendown: (1)

# patch_pre_layer = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")
# patch_pre_layer = tf.keras.models.Sequential(
#     [
#         tf.keras.layers.Conv2D(8, 3),
#         tf.keras.layers.MaxPool2D(),
#         tf_agents.keras_layers.inner_reshape.InnerReshape([None]*3, [-1]),
#     ]
# )
patch_pre_layer = tf_agents.keras_layers.inner_reshape.InnerReshape([None,None,3], [-1])
color_pre_layer = tf_agents.keras_layers.inner_reshape.InnerReshape([None], [-1])
motion_pre_layer = tf_agents.keras_layers.inner_reshape.InnerReshape([None], [-1])
preprocessing_layers = OrderedDict(
    [
        ("patch", patch_pre_layer),
        ("color", color_pre_layer),
        ("motion", motion_pre_layer),
    ]
)

def create_sequential_critic_net():
  value_layer_dict = {
          "patch": patch_pre_layer,
          "color": color_pre_layer,
          "motion": motion_pre_layer
  }
#   value_layer = sequential.Sequential([
#       value_layer_dict,
#       tf.keras.layers.Lambda(tf.nest.flatten),
#       tf.keras.layers.Concatenate(),
#       tf.keras.layers.Dense(1)])

  action_layer = tf.keras.layers.Dense(81)

  def sum_value_and_action_out(value_and_action_out):
    value_out_dict, action_out = value_and_action_out
    value_out = tf.concat(tf.nest.flatten(value_out_dict), axis=-1)
    # value_out = value_out_dict
    return tf.reshape(value_out + action_out, [1,-1])

  return sequential.Sequential([
      nest_map.NestMap((value_layer_dict, action_layer)),
      tf.keras.layers.Lambda(sum_value_and_action_out),
      tf.keras.layers.Dense(1)
  ])

preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

# with strategy.scope():
#     critic_net = PaintingCriticNetwork(
#         (observation_spec, action_spec),
#         observation_fc_layer_params=None,
#         action_fc_layer_params=None,
#         joint_fc_layer_params=critic_joint_fc_layer_params,
#         kernel_initializer="glorot_uniform",
#         last_kernel_initializer="glorot_uniform",
#         preprocessing_layers=preprocessing_layers,
#         preprocessing_combiner=preprocessing_combiner,
#     )

with strategy.scope():
    actor_net = PaintingActorNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
    )

with strategy.scope():
    train_step = train_utils.create_train_step()
    tf_agent = sac_agent.SacAgent(
          time_step_spec,
          action_spec,
          actor_network=actor_net,
          critic_network=create_sequential_critic_net(),
          actor_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=actor_learning_rate),
          critic_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=critic_learning_rate),
          alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=alpha_learning_rate),
          target_update_tau=target_update_tau,
          target_update_period=target_update_period,
          td_errors_loss_fn=tf.math.squared_difference,
          gamma=gamma,
          reward_scale_factor=reward_scale_factor,
          train_step_counter=train_step)
    # tf_agent = sac_agent.SacAgent(
    #     time_step_spec,
    #     action_spec,
    #     critic_network=create_sequential_critic_net(),
    #     actor_network=None,
    #     actor_optimizer=None,
    #     critic_optimizer=None,
    #     alpha_optimizer=None,
    # )
    # tf_agent = ddpg_agent.DdpgAgent(
    #     time_step_spec,
    #     action_spec,
    #     actor_network=actor_net,
    #     critic_network=critic_net,
    #     actor_optimizer=tf.compat.v1.train.AdamOptimizer(
    #         learning_rate=actor_learning_rate
    #     ),
    #     critic_optimizer=tf.compat.v1.train.AdamOptimizer(
    #         learning_rate=critic_learning_rate
    #     ),
    #     target_update_tau=target_update_tau,
    #     target_update_period=target_update_period,
    #     td_errors_loss_fn=tf.math.squared_difference,
    #     gamma=gamma,
    #     reward_scale_factor=reward_scale_factor,
    #     train_step_counter=train_step,
    # )

    tf_agent.initialize()

table_name = "uniform_table"
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
)

reverb_server = reverb.Server([table], port=reverb_port)

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server,
)

dataset = reverb_replay.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(
    50
)
experience_dataset_fn = lambda: dataset

tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_collect_policy, use_tf_function=True
)

random_policy = random_py_policy.RandomPyPolicy(
    collect_env.time_step_spec(), collect_env.action_spec()
)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    reverb_replay.py_client, table_name, sequence_length=2, stride_length=1
)

initial_collect_actor = actor.Actor(
    collect_env,
    random_policy,
    train_step,
    steps_per_run=initial_collect_steps,
    observers=[rb_observer],
)
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
    collect_env,
    collect_policy,
    train_step,
    steps_per_run=1,
    metrics=actor.collect_metrics(10),
    summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    observers=[rb_observer, env_step_metric],
)

eval_actor = actor.Actor(
    eval_env,
    eval_policy,
    train_step,
    episodes_per_run=num_eval_episodes,
    metrics=actor.eval_metrics(num_eval_episodes),
    summary_dir=os.path.join(tempdir, "eval"),
)

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir, tf_agent, train_step, interval=policy_save_interval
    ),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
    tempdir, train_step, tf_agent, experience_dataset_fn, triggers=learning_triggers
)


def get_eval_metrics():
    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
        results[metric.name] = metric.result()
    return results


metrics = get_eval_metrics()


def log_eval_metrics(step, metrics):
    eval_results = (", ").join(
        "{} = {:.6f}".format(name, result) for name, result in metrics.items()
    )
    print("step = {0}: {1}".format(step, eval_results))


log_eval_metrics(0, metrics)


# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

for iteration in range(num_iterations):
    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # Evaluating.
    step = agent_learner.train_step_numpy

    if eval_interval and step % eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        returns.append(metrics["AverageReturn"])

    if log_interval and step % log_interval == 0:
        print("step = {0}: loss = {1}".format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

# @test {"skip": true}

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel("Average Return")
plt.xlabel("Step")
plt.ylim()

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)

num_episodes = 1
video_filename = "sac_minitaur.mp4"
with imageio.get_writer(video_filename, fps=60) as video:
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        video.append_data(eval_env.render())
        while not time_step.is_last():
            action_step = eval_actor.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(eval_env.render())
embed_mp4(video_filename)
