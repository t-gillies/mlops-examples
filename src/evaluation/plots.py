from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    """Save a confusion-matrix heatmap to *out_path*."""
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    """Save an ROC curve plot to *out_path*."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    """Save a precision-recall curve plot to *out_path*."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Save a horizontal bar chart of the top *top_n* feature importances."""
    order = np.argsort(importances)[::-1]
    top_idx = order[:top_n]
    fig = plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
    )
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
