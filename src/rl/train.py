import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env  # noqa: F401

TRAIN = True

if __name__ == "__main__":
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    obs, info = env.reset()

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/",
    )

    if TRAIN:
        model.learn(total_timesteps=int(2e5))
        model.save("highway_dqn2/model")
        del model

    model = DQN.load("highway_dqn2/model", env=env)
    env = RecordVideo(
        env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15 # type: ignore
    env.unwrapped.set_record_video_wrapper(env) # type: ignore

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()