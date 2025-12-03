import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import numpy as np
from tqdm import tqdm

MODEL_PATH = "models/dqn_v2.zip"

NUM_EPISODES = 100
MAX_STEPS = 500

def main():
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config={
        "traffic_density": 1.2,
        "duration": 60
    })

    print(f"Carregando modelo: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH, env=env)

    episode_rewards = []
    episode_lengths = []

    collision_count = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    print(f"Rodando avaliação com {NUM_EPISODES} episódios...")

    for ep in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        done = truncated = False

        total_reward = 0
        steps = 0
        crashed = False

        while not (done or truncated) and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            action_counts[action] += 1

            # highway-env indica colisão via "crashed"
            if "crashed" in info and info["crashed"]:
                crashed = True

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if crashed:
            collision_count += 1

    env.close()

    # Resultados
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    print("\n=== MÉTRICAS DO MODELO ===")
    print(f"Episódios avaliados: {NUM_EPISODES}")
    print(f"Recompensa média: {episode_rewards.mean():.2f}")
    print(f"Desvio padrão da recompensa: {episode_rewards.std():.2f}")
    print(f"Duração média dos episódios: {episode_lengths.mean():.2f} steps")
    print(f"Taxa de colisão: {(collision_count / NUM_EPISODES) * 100:.2f}%")

    print("\nDistribuição de ações:")
    total_actions = sum(action_counts.values())
    for k, v in action_counts.items():
        print(f"Ação {k}: {v} ({v/total_actions:.1%})")
