import os
import csv
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import cv2
from tqdm import tqdm

DATASET_DIR = "dataset_big_highway"
CSV_PATH = os.path.join(DATASET_DIR, "dataset_highway_200k.csv")
MODEL_PATH = "models/dqn_v2.zip"

NUM_EPISODES = 500  
MAX_STEPS = 500     

os.makedirs(DATASET_DIR, exist_ok=True)

env = gym.make("highway-fast-v0", render_mode="rgb_array", config={
    "traffic_density": 1.2,
    "duration": 60 
})

print(f"Carregando Expert: {MODEL_PATH}")
model = DQN.load(MODEL_PATH, env=env)

action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

print(f"Iniciando geração de {NUM_EPISODES} episódios...")

# Abre o arquivo CSV uma única vez
with open(CSV_PATH, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    header = ["image_path", "action"]
    writer.writerow(header)

    for ep in tqdm(range(NUM_EPISODES), desc="Gerando Episódios"):
        
        ep_dir = os.path.join(DATASET_DIR, f"episode_{ep:04}") # 0000
        os.makedirs(ep_dir, exist_ok=True)

        obs, _ = env.reset()
        done = truncated = False
        step = 0

        while not (done or truncated) and step < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # type: ignore
            
            img_filename = f"{step:05}.png"
            img_full_path = os.path.join(ep_dir, img_filename)
            
            cv2.imwrite(img_full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            rel_path = os.path.join(f"episode_{ep:04}", img_filename)
            writer.writerow([rel_path, action])

            action_counts[action] += 1

            obs, reward, done, truncated, info = env.step(action)
            step += 1

env.close()

print("\n=== Estatísticas do Dataset Gerado ===")
total_frames = sum(action_counts.values())
print(f"Total de Frames: {total_frames}")
print("Distribuição de Ações:")
print(f"0 (LANE_LEFT): {action_counts[0]} ({action_counts[0]/total_frames:.1%})")
print(f"1 (IDLE):      {action_counts[1]} ({action_counts[1]/total_frames:.1%})")
print(f"2 (LANE_RIGHT): {action_counts[2]} ({action_counts[2]/total_frames:.1%})")
print(f"3 (FASTER):    {action_counts[3]} ({action_counts[3]/total_frames:.1%})")
print(f"4 (SLOWER):    {action_counts[4]} ({action_counts[4]/total_frames:.1%})")
print(f"CSV salvo em: {CSV_PATH}")