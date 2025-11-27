import os
import csv
import numpy as np
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import cv2

# =============== CONFIG ===============
DATASET_DIR = "dataset"
CSV_PATH = "dataset_highway.csv"
MODEL_PATH = "models/dqn_v2.zip"
NUM_EPISODES = 40
MAX_STEPS = 500
# =======================================

os.makedirs(DATASET_DIR, exist_ok=True)

env = gym.make("highway-fast-v0", render_mode="rgb_array")
model = DQN.load(MODEL_PATH, env=env)

with open(CSV_PATH, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Cabeçalho
    header = ["image_path"] + [f"f{i}" for i in range(25)] + ["action"]
    writer.writerow(header)

    for ep in range(NUM_EPISODES):
        print(f"\nEpisode {ep}")

        ep_dir = os.path.join(DATASET_DIR, f"episode_{ep:03}")
        frames_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        obs, _ = env.reset()
        done = truncated = False
        step = 0

        while not (done or truncated) and step < MAX_STEPS:
            # Ação do expert
            action, _ = model.predict(obs, deterministic=True)

            # Renderiza e salva frame
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # type: ignore
            img_path = f"{frames_dir}/{step:05}.png"
            cv2.imwrite(img_path, frame)

            # Flatten obs
            flat_obs = obs.flatten().tolist()

            # Salva linha no CSV
            row = [img_path] + flat_obs + [int(action)]
            writer.writerow(row)

            # Step no ambiente
            obs, reward, done, truncated, info = env.step(action)
            step += 1

env.close()
print("\nCSV gerado com sucesso!")
