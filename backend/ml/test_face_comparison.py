import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

doc_path = r'backend/media/documents/2026/09/27/c77a3e27e14d4eeebebd6871709f2808.jpg'
selfie_path = r'backend/media/selfies/2026/09/27/f74719ef8caa47358c3b82fef91d75fb.jpg'

doc_img = cv2.imread(doc_path)
selfie_img = cv2.imread(selfie_path)

# Crop faces
doc_face = doc_img[3010:3010+990, 2436:2436+563]
selfie_face = selfie_img[187:187+293, 252:252+224]

print("doc_face shape:", doc_face.shape)
print("selfie_face shape:", selfie_face.shape)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

t_doc = transform(Image.fromarray(cv2.cvtColor(doc_face, cv2.COLOR_BGR2RGB))).unsqueeze(0)
t_selfie = transform(Image.fromarray(cv2.cvtColor(selfie_face, cv2.COLOR_BGR2RGB))).unsqueeze(0)

with torch.no_grad():
    f_doc = model(t_doc).numpy()[0]
    f_selfie = model(t_selfie).numpy()[0]
    f_doc = f_doc / np.linalg.norm(f_doc)
    f_selfie = f_selfie / np.linalg.norm(f_selfie)

dot = float(np.dot(f_doc, f_selfie))
sim = (dot + 1.0) / 2.0
print(f"Cosine Dot: {dot:.4f}, Normalized Similarity: {sim * 100:.1f}%")
