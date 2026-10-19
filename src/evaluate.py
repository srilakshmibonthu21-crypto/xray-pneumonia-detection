"""
Clinical evaluation module for VGG16 Chest X-ray Pneumonia Detection.
Computes:
1. Confusion Matrix (TP, TN, FP, FN)
2. Clinical Sensitivity (Recall / True Positive Rate)
3. Clinical Specificity (True Negative Rate)
4. Precision (Positive Predictive Value)
5. F1-Score
6. ROC Curve & AUC Score
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
import tensorflow as tf
from src import config
from src.dataset import get_data_generators

def evaluate_model(model_path=None):
    if model_path is None:
        if config.BEST_MODEL_PATH.exists():
            model_path = config.BEST_MODEL_PATH
        elif config.FINAL_MODEL_PATH.exists():
            model_path = config.FINAL_MODEL_PATH
        else:
            raise FileNotFoundError(f"No trained model found at {config.BEST_MODEL_PATH} or {config.FINAL_MODEL_PATH}. Run train.py first.")

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    _, _, test_gen = get_data_generators()
    if test_gen is None or test_gen.samples == 0:
        raise ValueError("No test data found in data/test directory.")

    print(f"[INFO] Evaluating on {test_gen.samples} test images...")
    y_true = test_gen.classes
    test_gen.reset()

    # Predict probabilities
    y_pred_probs = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 2. ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    metrics_summary = {
        "Accuracy": float(accuracy),
        "Sensitivity (Recall)": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "True Positives (TP)": int(tp),
        "True Negatives (TN)": int(tn),
        "False Positives (FP)": int(fp),
        "False Negatives (FN)": int(fn)
    }

    # Save metrics to JSON
    metrics_path = config.RESULTS_DIR / "clinical_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[INFO] Metrics saved to: {metrics_path}")

    # Plot Confusion Matrix and ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Normal vs. Pneumonia)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # ROC Curve
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1].set_title("Receiver Operating Characteristic (ROC)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    cm_path = config.RESULTS_DIR / "evaluation_report.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Evaluation plots saved to: {cm_path}")

    print("\n" + "=" * 50)
    print("CLINICAL PERFORMANCE SUMMARY:")
    for k, v in metrics_summary.items():
        print(f"  {k:25s}: {v if isinstance(v, int) else f'{v:.4f}'}")
    print("=" * 50)

    return metrics_summary

if __name__ == "__main__":
    evaluate_model()
