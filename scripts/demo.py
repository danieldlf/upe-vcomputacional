import gymnasium as gym
import highway_env
import torch
import numpy as np
import cv2 # Usaremos OpenCV para mostrar a tela, já que pegaremos o array RGB
from src.vlm import MultimodalPolicy
from src.encoder import CLIPVisualEncoder

MODEL_PATH = "vlm_v3.pth" # Seu modelo treinado
MODEL_NAME = "Qwen/Qwen3-0.6B"
CLIP_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Dispositivo: {DEVICE}")
    visual_encoder = CLIPVisualEncoder(device=DEVICE)
    
    policy = MultimodalPolicy(
        model_name=MODEL_NAME, 
        clip_dim=CLIP_DIM, 
        action_size=5
    ).to(DEVICE)
    
    print(f"Carregando pesos de: {MODEL_PATH}")
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    policy.eval()

    env = gym.make("highway-v0", render_mode="rgb_array")
    while True:
        obs, info = env.reset()
        done = truncated = False
        
        while not (done or truncated):
            frame = env.render()
            clip_embedding = visual_encoder.encode_np(frame) # Shape: [1, 512] # type: ignore

            with torch.no_grad():
                logits = policy(clip_feat=clip_embedding)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, done, truncated, info = env.step(action)

            cv2_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # type: ignore

            action_map = {0: "LANE_L", 1: "IDLE", 2: "LANE_R", 3: "FASTER", 4: "SLOWER"}
            cv2.putText(cv2_img, f"Action: {action_map.get(action, action)}", (10, 30),  # type: ignore
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("VLM Autonomous Driving", cv2_img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()