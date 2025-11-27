import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPVisualEncoder:
    def __init__(self, device = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device) # type: ignore
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.model.eval()

    def encode_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device) # type: ignore

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_pil(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device) # type: ignore

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_np(self, image: np.ndarray):
        image = Image.fromarray(image).convert("RGB") # type: ignore
        return self.encode_pil(image) # type: ignore