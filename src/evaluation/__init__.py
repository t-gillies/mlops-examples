"""Model evaluation: metrics computation and diagnostic plots."""

from src.evaluation.metrics import compute_confusion_matrix, compute_metrics
from src.evaluation.plots import (
    plot_confusion,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)

__all__ = [
    "compute_confusion_matrix",
    "compute_metrics",
    "plot_confusion",
    "plot_feature_importance",
    "plot_pr_curve",
    "plot_roc_curve",
]
