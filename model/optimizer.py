"""
ML Loss Optimization — Core Optimizer Module
=============================================
Implements Logistic Regression via two optimization strategies:
  1. Gradient Descent  — first-order iterative method
  2. Newton's Method   — second-order method using the Hessian

Dataset: Breast Cancer Wisconsin (Diagnostic)
Task: Binary classification — Benign (0) vs Malignant (1)

Author: [Your Name]
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Sigmoid & Loss ────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: clips z to avoid overflow."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                          weights: np.ndarray, lam: float = 0.0) -> float:
    """
    Binary cross-entropy loss with optional L2 regularization.

    Args:
        y_true : Ground truth labels (0 or 1), shape (n,)
        y_pred : Predicted probabilities,       shape (n,)
        weights: Model weights for regularization, shape (d,)
        lam    : L2 regularization strength (lambda)

    Returns:
        Scalar loss value.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = len(y_true)
    ce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    reg = (lam / (2 * n)) * np.sum(weights[1:] ** 2)  # skip bias
    return ce + reg


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Holds the complete history of a training run."""
    method: str
    losses: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    training_time_ms: float = 0.0
    final_weights: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "losses": [round(l, 6) for l in self.losses],
            "accuracies": [round(a, 4) for a in self.accuracies],
            "iterations": self.iterations,
            "converged": self.converged,
            "training_time_ms": round(self.training_time_ms, 2),
            "final_loss": round(self.losses[-1], 6) if self.losses else None,
            "final_accuracy": round(self.accuracies[-1], 4) if self.accuracies else None,
        }


# ── Base Optimizer ────────────────────────────────────────────────────────────

class LogisticBase:
    """
    Abstract base for logistic regression optimizers.

    Attributes:
        weights : Learned weight vector (including bias at index 0)
        tol     : Convergence tolerance (early stopping)
        lam     : L2 regularization coefficient
    """

    def __init__(self, tol: float = 1e-6, lam: float = 0.0):
        self.tol = tol
        self.lam = lam
        self.weights: Optional[np.ndarray] = None

    def _init_weights(self, n_features: int):
        """Xavier-style initialization for logistic regression."""
        limit = np.sqrt(1.0 / n_features)
        self.weights = np.random.uniform(-limit, limit, size=n_features + 1)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones for the bias term."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities (requires fit first)."""
        X_b = self._add_bias(X)
        return sigmoid(X_b @ self.weights)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ── Gradient Descent Optimizer ────────────────────────────────────────────────

class GradientDescentOptimizer(LogisticBase):
    """
    Logistic Regression via Batch Gradient Descent.

    Update rule:
        w ← w − α * ∇L(w)
    where
        ∇L(w) = (1/n) * Xᵀ(ŷ − y) + (λ/n) * w  [L2 regularized]

    Converges in O(1/ε) iterations; learning rate α is critical.
    """

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000,
                 tol: float = 1e-6, lam: float = 0.0):
        super().__init__(tol=tol, lam=lam)
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Fit the model using gradient descent.

        Args:
            X, y       : Training features and labels
            X_val, y_val: Optional validation set for loss tracking

        Returns:
            OptimizationResult with full loss/accuracy history.
        """
        n, d = X.shape
        X_b = self._add_bias(X)
        self._init_weights(d)

        result = OptimizationResult(method="Gradient Descent")
        start = time.perf_counter()

        for i in range(self.max_iter):
            y_hat = sigmoid(X_b @ self.weights)
            error = y_hat - y

            # Gradient with L2 regularization (skip bias term)
            grad = (X_b.T @ error) / n
            reg_term = np.zeros_like(self.weights)
            reg_term[1:] = (self.lam / n) * self.weights[1:]
            grad += reg_term

            self.weights -= self.lr * grad

            # Track metrics
            loss = binary_cross_entropy(y, y_hat, self.weights, self.lam)
            acc = self.accuracy(X, y)
            result.losses.append(loss)
            result.accuracies.append(acc)

            # Convergence check
            if i > 0 and abs(result.losses[-2] - loss) < self.tol:
                result.converged = True
                result.iterations = i + 1
                break
        else:
            result.iterations = self.max_iter

        result.training_time_ms = (time.perf_counter() - start) * 1000
        result.final_weights = self.weights.copy()
        return result


# ── Newton's Method Optimizer ─────────────────────────────────────────────────

class NewtonMethodOptimizer(LogisticBase):
    """
    Logistic Regression via Newton–Raphson Method.

    Update rule:
        w ← w − H⁻¹ * ∇L(w)
    where
        H = (1/n) * Xᵀ * diag(ŷ(1−ŷ)) * X  (Hessian of cross-entropy)

    Converges quadratically — typically 5–20 iterations.
    Hessian is regularized (+ λI) for numerical stability.
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6, lam: float = 1e-4):
        super().__init__(tol=tol, lam=lam)
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Fit the model using Newton's Method.

        Args:
            X, y: Training features and labels.

        Returns:
            OptimizationResult with full loss/accuracy history.
        """
        n, d = X.shape
        X_b = self._add_bias(X)
        self._init_weights(d)

        result = OptimizationResult(method="Newton's Method")
        start = time.perf_counter()

        for i in range(self.max_iter):
            y_hat = sigmoid(X_b @ self.weights)
            error = y_hat - y

            # Gradient (score vector)
            grad = (X_b.T @ error) / n

            # Hessian: H = (1/n) * Xᵀ W X  where W = diag(ŷ(1−ŷ))
            W = y_hat * (1 - y_hat)
            H = (X_b.T * W) @ X_b / n

            # Tikhonov (L2) regularization on Hessian — avoids singularity
            H += self.lam * np.eye(H.shape[0])

            # Newton step: w ← w − H⁻¹ g
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                # Fallback: pseudo-inverse if Hessian is singular
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]

            self.weights -= delta

            loss = binary_cross_entropy(y, y_hat, self.weights, self.lam)
            acc = self.accuracy(X, y)
            result.losses.append(loss)
            result.accuracies.append(acc)

            if i > 0 and abs(result.losses[-2] - loss) < self.tol:
                result.converged = True
                result.iterations = i + 1
                break
        else:
            result.iterations = self.max_iter

        result.training_time_ms = (time.perf_counter() - start) * 1000
        result.final_weights = self.weights.copy()
        return result


# ── Quick sanity check (run directly) ────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    gd = GradientDescentOptimizer(learning_rate=0.1, max_iter=500)
    gd_result = gd.fit(X_train, y_train)
    print(f"GD  → iters={gd_result.iterations:>4d} | "
          f"loss={gd_result.losses[-1]:.4f} | "
          f"test acc={gd.accuracy(X_test, y_test):.4f} | "
          f"converged={gd_result.converged}")

    nm = NewtonMethodOptimizer(max_iter=50)
    nm_result = nm.fit(X_train, y_train)
    print(f"NM  → iters={nm_result.iterations:>4d} | "
          f"loss={nm_result.losses[-1]:.4f} | "
          f"test acc={nm.accuracy(X_test, y_test):.4f} | "
          f"converged={nm_result.converged}")