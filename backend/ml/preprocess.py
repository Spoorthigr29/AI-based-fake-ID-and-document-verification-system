"""
VerifyX AI - Image Preprocessing & Data Augmentation Pipeline
============================================================
Handles high-fidelity loading, validation, color space normalization,
resolution standardization (224x224), and subtle forensic data augmentation.
"""

import os
import random
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
from typing import Tuple, Optional, Union

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DocumentCompressionSimulation:
    """Simulates realistic JPEG compression artifacts and slight noise."""
    def __init__(self, quality_range=(65, 95), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            import io
            buffer = io.BytesIO()
            quality = random.randint(*self.quality_range)
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


class SubtleDocumentBlur:
    """Applies subtle optical or sensor blur."""
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(0.3, 1.0)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Data augmentation for training set:
    - Subtle rotation (-5 to +5 deg) to simulate document skew
    - Mild perspective scaling & resizing (preserving core structure)
    - Gentle color jitter (brightness, contrast, slight saturation)
    - JPEG compression simulation
    - Subtle blur simulation
    - Normalization with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=(-5, 5), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        DocumentCompressionSimulation(quality_range=(70, 95), p=0.35),
        SubtleDocumentBlur(p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic preprocessing for validation, testing, and production inference:
    - Standard RGB conversion
    - Resize to 224x224
    - Tensor conversion & ImageNet normalization
    - NO random augmentation
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_document_image(
    image_input: Union[str, Image.Image, np.ndarray],
    image_size: int = 224
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load, verify, and preprocess a single document image for model inference.
    Returns:
        (tensor_batch [1, 3, 224, 224], pil_image_rgb)
    """
    if hasattr(image_input, 'pil'):
        pil_img = image_input.pil
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Document image not found at: {image_input}")
        if image_input.lower().endswith('.pdf'):
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(image_input)
                try:
                    page = pdf[0]
                    pil_img = page.render(scale=2.0).to_pil().convert("RGB")
                finally:
                    pdf.close()
            except Exception:
                pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:
            pil_img = Image.fromarray(image_input[:, :, :3]).convert("RGB")
        else:
            # OpenCV BGR to RGB
            pil_img = Image.fromarray(image_input[:, :, ::-1]).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    transform = get_eval_transforms(image_size)
    tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor, pil_img
