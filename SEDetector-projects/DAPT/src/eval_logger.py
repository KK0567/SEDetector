# eval_logger.py
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(cm_norm: np.ndarray, class_names, out_png: str, title="Confusion Matrix (Normalized)"):
    plt.figure(figsize=(7, 7))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_png, dpi=300)
    plt.close()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names, out_dir: str, prefix: str):
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p = float(p); r = float(r); f1 = float(f1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    metrics = {
        "accuracy": acc,
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "support_total": int(len(y_true)),
    }

    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, f"{prefix}_metrics.json"), metrics)
    plot_confusion_matrix(
        cm_norm,
        class_names=class_names,
        out_png=os.path.join(out_dir, f"{prefix}_cm.png")
    )
    return metrics
