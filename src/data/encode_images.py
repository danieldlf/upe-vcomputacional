import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- CONFIG ---
BASE_DIR = "dataset_big_highway" 
CSV_PATH = os.path.join(BASE_DIR, "dataset_highway_200k.csv")
BATCH_SIZE = 64
NUM_WORKERS = 4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Carregando CLIP Processor...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- DATASET ---
class ImageListDataset(Dataset):
    def __init__(self, csv_path, base_dir):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]["image_path"]
        full_path = os.path.join(self.base_dir, rel_path)
        return full_path

    # O collate_fn precisa ser estático ou usar o processor global
    def collate_fn(self, batch):
        paths = batch
        images = []
        valid_paths = []
        
        for p in paths:
            try:
                # O processamento de imagem é rápido, o gargalo é I/O
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Erro ao abrir {p}: {e}")
                
        if not images:
            return None, None
            
        # O processor global é usado aqui pelos workers
        inputs = processor(images=images, return_tensors="pt", padding=True) # type: ignore
        return inputs, valid_paths

# --- FUNÇÃO MAIN ---
def main():
    print(f"Usando dispositivo: {DEVICE}")

    # Carregamos o modelo APENAS no processo principal e jogamos para a GPU
    print("Carregando CLIP Model para GPU...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE) # type: ignore
    model.eval()

    # Dataset e Loader
    dataset = ImageListDataset(CSV_PATH, BASE_DIR)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        collate_fn=dataset.collate_fn,
        pin_memory=True # Acelera transferência CPU -> GPU
    )

    print(f"Iniciando processamento de {len(dataset)} imagens...")

    # Loop de Processamento
    with torch.no_grad():
        for batch_inputs, paths in tqdm(loader):
            if batch_inputs is None: continue
            
            # Mover tensores para GPU
            inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}
            
            # Inferência
            features = model.get_image_features(**inputs)
            
            # Normalização
            features = features / features.norm(dim=-1, keepdim=True)
            
            # CPU e Numpy (float16)
            features_np = features.cpu().numpy().astype(np.float16)
            
            # Salvamento
            for i, path in enumerate(paths):
                save_path = os.path.splitext(path)[0] + ".npy"
                np.save(save_path, features_np[i])

    print("✅ Pré-processamento concluído! Arquivos .npy gerados.")

# --- PONTO DE ENTRADA (OBRIGATÓRIO NO WINDOWS) ---
if __name__ == "__main__":
    main()