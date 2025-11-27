import torch
import pandas as pd
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import tqdm

from src.vlm.model import MultimodalPolicy
from src.encoder import CLIPVisualEncoder

MODEL_PATH = "vlm_v3.pth"
CSV_PATH = "dataset_big_highway/dataset_highway_200k.csv"
ROOT_DIR = "dataset_big_highway"
NUM_SAMPLES = 1000 # Número de amostras para avaliar (0 = todas)
MODEL_NAME = "Qwen/Qwen3-0.6B"
CLIP_DIM = 512

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT", 
    3: "FASTER",
    4: "SLOWER"
}

def compute_confusion_matrix(predictions, ground_truth, num_classes=5):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, gt in zip(predictions, ground_truth):
        matrix[gt][pred] += 1
    return matrix

def print_confusion_matrix(matrix, action_names):
    print("\n📊 Matriz de Confusão:")
    print("     ", "  ".join([f"{action_names[i][:8]:>8}" for i in range(len(matrix))]))
    for i, row in enumerate(matrix):
        print(f"{action_names[i][:8]:>8}", "  ".join([f"{val:>8}" for val in row]))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Dispositivo: {device}")
    print(f"📁 Modelo: {MODEL_PATH}")
    print(f"📊 Dataset: {CSV_PATH}\n")

    df = pd.read_csv(CSV_PATH)
    
    if NUM_SAMPLES > 0 and NUM_SAMPLES < len(df):
        df = df.sample(NUM_SAMPLES, random_state=42)
        print(f"📦 Avaliando {NUM_SAMPLES} amostras aleatórias")
    else:
        print(f"📦 Avaliando {len(df)} amostras (dataset completo)")

    print("🧠 Carregando modelo VLM...")
    model = MultimodalPolicy(MODEL_NAME, CLIP_DIM, action_size=5).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erro: Modelo não encontrado em {MODEL_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("👁️ Carregando CLIP Encoder...")
    clip = CLIPVisualEncoder(device=device)

    predictions = []
    ground_truth = []
    correct = 0
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    top3_correct = 0

    print("\n🔄 Processando amostras...\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Avaliando"):
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(ROOT_DIR, img_path)
        
        action_gt = int(row["action"])

        if not os.path.exists(img_path):
            print(f"⚠️ Imagem não encontrada: {img_path}")
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"⚠️ Erro ao processar {img_path}: {e}")
            continue

        clip_feat = clip.encode_np(img).to(device)

        with torch.no_grad():
            logits = model(clip_feat=clip_feat)
            pred = torch.argmax(logits, dim=-1).item()

            top3 = torch.topk(logits, k=3, dim=-1).indices[0].cpu().numpy()
            if action_gt in top3:
                top3_correct += 1

        predictions.append(pred)
        ground_truth.append(action_gt)

        if pred == action_gt:
            correct += 1
            correct_per_class[action_gt] += 1
        
        total_per_class[action_gt] += 1

    total = len(predictions)
    
    if total == 0:
        print("❌ Nenhuma amostra válida processada!")
        return

    accuracy = correct / total
    top3_accuracy = top3_correct / total

    print("\n" + "="*50)
    print("📈 RESULTADOS DA AVALIAÇÃO")
    print("="*50)
    print(f"✅ Acurácia Geral: {accuracy:.2%} ({correct}/{total})")
    print(f"🎯 Top-3 Accuracy: {top3_accuracy:.2%}")
    
    print("\n📊 Acurácia por Classe:")
    for action_id in sorted(total_per_class.keys()):
        action_name = ACTION_NAMES[action_id]
        class_acc = correct_per_class[action_id] / total_per_class[action_id] if total_per_class[action_id] > 0 else 0
        print(f"  {action_name:12s}: {class_acc:.2%} ({correct_per_class[action_id]}/{total_per_class[action_id]} corretos)")

    conf_matrix = compute_confusion_matrix(predictions, ground_truth)
    print_confusion_matrix(conf_matrix, ACTION_NAMES)

    print("\n" + "="*50)
    print("✨ Avaliação concluída!")
    print("="*50)

if __name__ == "__main__":
    main()

