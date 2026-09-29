"""
VerifyX AI - Model Evaluation & Diagnostic Telemetry Pipeline
============================================================
Evaluates the trained Document Authenticity Model on the completely unseen,
held-out Test Set.

Generates:
1. Confusion Matrix Plot (reports/confusion_matrix.png)
2. Comprehensive Classification Report (reports/classification_report.txt)
3. False Positive & False Negative Category Breakdown
4. Precision, Recall, F1, ROC-AUC, and specific Fake-Document Recall metrics.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Dict, Any, List, Optional

from preprocess import get_eval_transforms, preprocess_document_image
from train_document_model import DocumentAuthenticityNet

CATEGORIES = ["pan", "aadhaar", "passport", "driving_license", "voter_id"]
REPORTS_DIR = "reports"
MODEL_PATH = "models/best_document_authenticity_model.pth"


def load_trained_model(model_path: str = MODEL_PATH, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[int, str]]:
    """Load trained authenticity model checkpoint from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        # Fallback to backend path if exists
        alt_path = os.path.join("backend", "ml_models", "best_document_authenticity_model.pth")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = DocumentAuthenticityNet(model_name=checkpoint.get("model_name", "efficientnet_b0"), num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_class = checkpoint.get("idx_to_class", {0: "fake", 1: "real"})
    return model, idx_to_class


def run_test_evaluation(test_dir: str = "dataset/test") -> Dict[str, Any]:
    """Execute evaluation on held-out test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"VERIFYX AI - HELD-OUT TEST DATASET EVALUATION")
    print(f"Device: {device}")
    print(f"========================================================")

    model, idx_to_class = load_trained_model(MODEL_PATH, device)
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    eval_tf = get_eval_transforms(224)
    test_ds = ImageFolder(test_dir, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    target_names = [idx_to_class[i].upper() for i in range(len(idx_to_class))]

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Specific Fake class index
    fake_idx = class_to_idx.get("fake", 0)
    real_idx = class_to_idx.get("real", 1)

    fake_precision = precision_score(all_targets == fake_idx, all_preds == fake_idx, zero_division=0)
    fake_recall = recall_score(all_targets == fake_idx, all_preds == fake_idx, zero_division=0)
    fake_f1 = f1_score(all_targets == fake_idx, all_preds == fake_idx, zero_division=0)

    try:
        roc_auc = roc_auc_score(all_targets == real_idx, all_probs[:, real_idx])
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(all_targets, all_preds)
    clf_rep_str = classification_report(all_targets, all_preds, target_names=target_names, digits=4)

    # False positive and false negative analysis
    fp = np.sum((all_preds == real_idx) & (all_targets == fake_idx))
    fn = np.sum((all_preds == fake_idx) & (all_targets == real_idx))

    # Detailed Category Breakdown on Test Samples
    cat_results = {cat: {"total": 0, "correct": 0, "real_correct": 0, "fake_correct": 0, "real_total": 0, "fake_total": 0} for cat in CATEGORIES}

    sample_eval_table = []

    for idx, (filepath, target_label) in enumerate(test_ds.samples):
        filename = os.path.basename(filepath)
        pred_label = all_preds[idx]
        pred_prob = all_probs[idx][pred_label]
        is_correct = (pred_label == target_label)

        # Categorize
        cat_found = "other"
        for cat in CATEGORIES:
            if cat in filename.lower():
                cat_found = cat
                break

        if cat_found in cat_results:
            cat_results[cat_found]["total"] += 1
            if is_correct:
                cat_results[cat_found]["correct"] += 1
            if target_label == real_idx:
                cat_results[cat_found]["real_total"] += 1
                if is_correct:
                    cat_results[cat_found]["real_correct"] += 1
            else:
                cat_results[cat_found]["fake_total"] += 1
                if is_correct:
                    cat_results[cat_found]["fake_correct"] += 1

        # Collect sample records for test table
        if len(sample_eval_table) < 25 or not is_correct:
            sample_eval_table.append({
                "image": filename,
                "category": cat_found.upper(),
                "actual": idx_to_class[target_label].upper(),
                "predicted": idx_to_class[pred_label].upper(),
                "confidence": f"{pred_prob * 100:.1f}%",
                "correct": "✓ PASS" if is_correct else "✗ FAIL"
            })

    # Plot Confusion Matrix
    plot_confusion_matrix(cm, target_names)

    # Save Classification Report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as rep_file:
        rep_file.write("=" * 65 + "\n")
        rep_file.write("VERIFYX AI - DOCUMENT AUTHENTICITY HELD-OUT TEST REPORT\n")
        rep_file.write("=" * 65 + "\n\n")
        rep_file.write(f"Total Test Images:        {len(all_targets)}\n")
        rep_file.write(f"Overall Accuracy:         {acc * 100:.2f}%\n")
        rep_file.write(f"Weighted Precision:       {prec_weighted:.4f}\n")
        rep_file.write(f"Weighted Recall:          {rec_weighted:.4f}\n")
        rep_file.write(f"Weighted F1 Score:        {f1_weighted:.4f}\n")
        rep_file.write(f"ROC-AUC Score:            {roc_auc:.4f}\n\n")
        rep_file.write("--- CRITICAL FAKE-DOCUMENT DETECTION METRICS ---\n")
        rep_file.write(f"Fake Document Precision:  {fake_precision * 100:.2f}%\n")
        rep_file.write(f"Fake Document Recall:     {fake_recall * 100:.2f}%\n")
        rep_file.write(f"Fake Document F1 Score:   {fake_f1 * 100:.2f}%\n")
        rep_file.write(f"False Positives (Fake predicted as Real): {fp}\n")
        rep_file.write(f"False Negatives (Real predicted as Fake): {fn}\n\n")
        rep_file.write("--- CLASSIFICATION REPORT ---\n")
        rep_file.write(clf_rep_str + "\n\n")
        rep_file.write("--- PER-CATEGORY ACCURACY BREAKDOWN ---\n")
        for cat, c_data in cat_results.items():
            cat_acc = (c_data["correct"] / max(1, c_data["total"])) * 100
            rep_file.write(f"{cat.upper().replace('_', ' ')}:\n")
            rep_file.write(f"  Overall Accuracy: {cat_acc:.1f}% ({c_data['correct']}/{c_data['total']})\n")
            rep_file.write(f"  Real Accurate:    {c_data['real_correct']}/{c_data['real_total']}\n")
            rep_file.write(f"  Fake Accurate:    {c_data['fake_correct']}/{c_data['fake_total']}\n\n")
        rep_file.write("=" * 65 + "\n")

    print(f"\nSaved held-out test classification report to: {report_path}")

    return {
        "accuracy": acc,
        "precision": prec_weighted,
        "recall": rec_weighted,
        "f1": f1_weighted,
        "roc_auc": roc_auc,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": cm,
        "category_results": cat_results,
        "sample_eval_table": sample_eval_table,
        "classification_report_str": clf_rep_str
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Generate and save formatted confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        annot_kws={"size": 16, "weight": "bold"}
    )
    plt.title("Document Authenticity Confusion Matrix\n(Held-out Unseen Test Set)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.ylabel("Actual True Label", fontsize=11, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    results = run_test_evaluation()
    print("\n--- Summary of Test Evaluation ---")
    print(f"Accuracy:        {results['accuracy']*100:.2f}%")
    print(f"F1 Score:        {results['f1']:.4f}")
    print(f"Fake Recall:     {results['fake_recall']*100:.2f}%")
    print(f"ROC-AUC:         {results['roc_auc']:.4f}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
