import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Test deep face embedder
class DeepFaceEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use pretrained ResNet18 as deep feature extractor
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        # Remove final classification layer to get 512-d feature embedding
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(tensor).cpu().numpy()[0]
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
        return feat

    def compute_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        dot = float(np.dot(emb_a, emb_b))
        # Map cosine similarity from [-1, 1] to [0.0, 1.0]
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))

embedder = DeepFaceEmbedder()
print("DeepFaceEmbedder initialized successfully!")
