from gym.envs.registration import register

register(
    id='Painter-v0',
    entry_point='gym_painting.envs:PaintingEnv',
    max_episode_steps=100000,
    reward_threshold=1000
)
