import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import time

from scripts.demo import main

MODEL_PATH = "models/dqn_v2.zip"

def main():
    # Cria ambiente com renderização na tela
    env = gym.make("highway-fast-v0", render_mode="human", config={
        "traffic_density": 1.2,
        "duration": 60
    })

    print(f"Carregando modelo: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH, env=env)

    obs, _ = env.reset()

    print("Iniciando DEMO... Ctrl+C para sair.")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            env.render()

            # Controla velocidade da simulação
            time.sleep(0.03)

            if done or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nDemo finalizada pelo usuário.")

    env.close()
