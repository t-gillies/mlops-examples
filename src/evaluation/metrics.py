import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute the standard evaluation metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Keys: ``val_accuracy``, ``val_f1_macro``, ``val_precision``,
        ``val_recall``, ``val_roc_auc``, ``val_pr_auc``.
    """
    return {
        "val_accuracy": accuracy_score(y_true, y_pred),
        "val_f1_macro": f1_score(y_true, y_pred, average="macro"),
        "val_precision": precision_score(y_true, y_pred),
        "val_recall": recall_score(y_true, y_pred),
        "val_roc_auc": roc_auc_score(y_true, y_proba),
        "val_pr_auc": average_precision_score(y_true, y_proba),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return the confusion matrix for *y_true* vs *y_pred*."""
    return confusion_matrix(y_true, y_pred)
