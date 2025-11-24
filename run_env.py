import gymnasium as gym
import highway_env

env = gym.make('highway-v0', render_mode='human')
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()