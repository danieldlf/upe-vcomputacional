from stable_baselines3 import DQN
import gymnasium as gym
import highway_env

env = gym.make("highway-fast-v0", render_mode="human")

model = DQN.load("models/dqn_v2.zip", env=env)
obs, _ = env.reset()

while True:
    done = truncated = False
    obs, info = env.reset()

    print(obs)

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()