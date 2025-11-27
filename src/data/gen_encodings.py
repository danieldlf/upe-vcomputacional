import numpy as np
from PIL import Image
from src.encoder import CLIPVisualEncoder
import csv

def gen_encodings(dataset_path):
    encoder = CLIPVisualEncoder()
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            img_path = r["image_path"]
            feat = encoder.encode_pil(Image.open(img_path).convert("RGB"))  # [1, clip_dim]
            feat = feat.cpu().numpy().squeeze()
            np.save(img_path.replace(".png", ".npy"), feat)