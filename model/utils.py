"""
ML Loss Optimization — Data & Metrics Utilities
================================================
Handles dataset loading, preprocessing pipeline, and evaluation metrics
for the Breast Cancer classification task.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score
)


# ── Data Pipeline ─────────────────────────────────────────────────────────────

def load_and_split(test_size: float = 0.2, random_state: int = 42):
    """
    Load Breast Cancer dataset, standardize, and split.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y        # preserve class balance
    )
    return X_train, X_test, y_train, y_test, scaler, feature_names


def get_dataset_info() -> dict:
    """Return dataset summary for the dashboard."""
    data = load_breast_cancer()
    unique, counts = np.unique(data.target, return_counts=True)
    return {
        "n_samples": int(data.data.shape[0]),
        "n_features": int(data.data.shape[1]),
        "classes": data.target_names.tolist(),
        "class_distribution": {
            data.target_names[int(k)]: int(v)
            for k, v in zip(unique, counts)
        },
        "feature_names": data.feature_names.tolist(),
    }


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Comprehensive evaluation of a fitted optimizer model.

    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC-ROC,
        confusion matrix, and per-class metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True
    )

    return {
        "accuracy": round(float(np.mean(y_pred == y_test)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "Benign": {
                "precision": round(report["Benign"]["precision"], 4),
                "recall": round(report["Benign"]["recall"], 4),
                "f1": round(report["Benign"]["f1-score"], 4),
            },
            "Malignant": {
                "precision": round(report["Malignant"]["precision"], 4),
                "recall": round(report["Malignant"]["recall"], 4),
                "f1": round(report["Malignant"]["f1-score"], 4),
            },
        }
    }