"""
VerifyX AI - Document Authenticity Deep Learning Model Training Pipeline
========================================================================
Trains a PyTorch Transfer Learning model (EfficientNet-B0 / ResNet50) for
binary document authenticity detection (REAL vs FAKE/TAMPERED).

Features:
- Two-phase training: Head warm-up followed by fine-tuning of upper backbone layers.
- AdamW optimizer with CosineAnnealingLR / ReduceLROnPlateau scheduler.
- Comprehensive metric tracking: Loss, Accuracy, Precision, Recall, F1, ROC-AUC.
- Early stopping on Validation F1/ROC-AUC.
- Saves best checkpoint to models/best_document_authenticity_model.pth and
  backend/ml_models/best_document_authenticity_model.pth.
- Generates training history curve report saved to reports/training_history.png.
"""

import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Any, List, Optional

# Import custom transforms
try:
    from ml.preprocess import get_train_transforms, get_eval_transforms
except ImportError:
    try:
        from .preprocess import get_train_transforms, get_eval_transforms
    except ImportError:
        from preprocess import get_train_transforms, get_eval_transforms

MODEL_SAVE_PATHS = [
    "models/best_document_authenticity_model.pth",
    "backend/ml_models/best_document_authenticity_model.pth"
]
REPORTS_DIR = "reports"


class DocumentAuthenticityNet(nn.Module):
    """Transfer learning architecture for document authenticity screening."""
    def __init__(self, model_name: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name

        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, 128),
                nn.SiLU(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(128, num_classes)
            )
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model backbone: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze head
        if hasattr(self.backbone, 'classifier'):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def unfreeze_top_layers(self, num_blocks: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = True


def load_dataset_folders(data_dir: str = "dataset") -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Load train, val, and test ImageFolder datasets with appropriate transforms."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    train_tf = get_train_transforms(224)
    eval_tf = get_eval_transforms(224)

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=eval_tf)
    test_ds = ImageFolder(test_dir, transform=eval_tf)

    # Class mappings (e.g. {'fake': 0, 'real': 1} or vice versa)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Dataset Loaded Successfully:")
    print(f"  Classes: {class_to_idx}")
    print(f"  Train samples:      {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Test samples:       {len(test_ds)}")

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, idx_to_class


def evaluate_dataset(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Compute evaluation metrics across a data loader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            # Probability for positive class (real / fake)
            all_probs.extend(probs[:, 1].cpu().numpy())

    total_samples = len(all_targets)
    epoch_loss = running_loss / max(1, total_samples)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = 0.5

    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }


def train_authenticity_model(
    data_dir: str = "dataset",
    max_epochs: int = 6,
    patience: int = 3,
    device_name: str = "auto"
) -> Dict[str, Any]:
    """Execute complete 2-phase transfer learning training loop with early stopping."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"\n========================================================", flush=True)
    print(f"VERIFYX AI - TRAINING DOCUMENT AUTHENTICITY MODEL", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"========================================================", flush=True)

    train_loader, val_loader, test_loader, idx_to_class = load_dataset_folders(data_dir)

    # Initialize model
    model = DocumentAuthenticityNet(model_name="efficientnet_b0", num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [],
        "val_f1": [], "val_roc_auc": []
    }

    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    # Phase 1: Train classification head (epochs 1 to 2)
    print("\n[Phase 1] Training classification head with frozen backbone...", flush=True)
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

    for epoch in range(1, max_epochs + 1):
        if epoch == 3:
            print("\n[Phase 2] Unfreezing backbone for fine-tuning...", flush=True)
            model.unfreeze_top_layers()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - 2, eta_min=1e-6)

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        if epoch >= 3 and scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # Validation Step
        val_metrics = evaluate_dataset(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Val Prec: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}",
            flush=True
        )

        # Early Stopping and Checkpointing on Val F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience and epoch >= 6:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).")
                break

    # Restore best weights and save
    model.load_state_dict(best_model_wts)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "efficientnet_b0",
        "num_classes": 2,
        "idx_to_class": idx_to_class,
        "best_val_f1": best_val_f1,
        "history": history
    }

    for path in MODEL_SAVE_PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved best model checkpoint to: {path}")

    # Plot & Save Training History
    plot_training_curves(history)

    return {
        "model": model,
        "test_loader": test_loader,
        "idx_to_class": idx_to_class,
        "history": history
    }


def plot_training_curves(history: Dict[str, List[float]]):
    """Generate and save training history plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#0875E1", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", color="#F59E0B", lw=2)
    axes[0].set_title("Training and Validation Loss", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy & F1 curve
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy (%)", color="#10B981", lw=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Accuracy (%)", color="#6366F1", lw=2)
    axes[1].plot(epochs, [f * 100 for f in history["val_f1"]], label="Validation F1 (%)", color="#EF4444", lw=2, ls='--')
    axes[1].set_title("Accuracy & F1 Metric Progression", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "training_history.png")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training history chart to: {plot_path}")


if __name__ == "__main__":
    train_authenticity_model()
