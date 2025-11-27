import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class DrivingDataset(Dataset):
    def __init__(self, csv_path, root_dir="dataset_big_highway"):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_rel_path = row["image_path"]
        
        npy_rel_path = os.path.splitext(img_rel_path)[0] + ".npy"
        npy_full_path = os.path.join(self.root_dir, npy_rel_path)
        
        try:
            clip_feat = np.load(npy_full_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo .npy não encontrado: {npy_full_path}. Rode o script de preprocessamento primeiro!")

        clip_feat = torch.from_numpy(clip_feat).float()
        action = int(row["action"])
        
        return {
            "clip_feat": clip_feat,
            "action": torch.tensor(action, dtype=torch.long)
        }