"""
Phase 1 – Linear Algebra Core
PCA on features including distance to HOD/LOD, session time as sine/cosine, candle stats.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class CovarianceMatrix:
    """Compute and manage covariance matrices for feature sets."""

    @staticmethod
    def compute(X: NDArray[np.float64], bias: bool = False) -> NDArray[np.float64]:
        return np.cov(X, rowvar=False, bias=bias)

    @staticmethod
    def correlation(X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.corrcoef(X, rowvar=False)


class MatrixOps:
    """Matrix inverse with Cholesky fallback, Gram-Schmidt, SVD."""

    @staticmethod
    def safe_inverse(A: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            L = np.linalg.cholesky(A)
            L_inv = np.linalg.inv(L)
            return L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    @staticmethod
    def gram_schmidt(V: NDArray[np.float64]) -> NDArray[np.float64]:
        n, k = V.shape
        U = np.zeros_like(V, dtype=np.float64)
        for i in range(k):
            u = V[:, i].copy()
            for j in range(i):
                u -= np.dot(U[:, j], V[:, i]) * U[:, j]
            norm = np.linalg.norm(u)
            U[:, i] = u / norm if norm > 1e-12 else u
        return U

    @staticmethod
    def svd(X: NDArray[np.float64]) -> Tuple[NDArray, NDArray, NDArray]:
        return np.linalg.svd(X, full_matrices=False)


class OLSRegression:
    """
    Gauss-Markov OLS: β = (X'X)^{-1} X'y
    Under Gauss-Markov conditions, OLS is BLUE.
    """

    def __init__(self) -> None:
        self.beta: NDArray[np.float64] | None = None
        self.residuals: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "OLSRegression":
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack([ones, X])
        XtX_inv = MatrixOps.safe_inverse(X_aug.T @ X_aug)
        self.beta = XtX_inv @ X_aug.T @ y
        self.residuals = y - X_aug @ self.beta
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack([ones, X])
        return X_aug @ self.beta

    @property
    def r_squared(self) -> float:
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.residuals + self.predict(np.zeros((len(self.residuals), self.beta.shape[0] - 1)))) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class PCAReducer:
    """
    Full PCA feature reducer.
    Input: matrix of indicators (rows=observations, cols=features)
    Output: reduced feature matrix retaining variance_threshold of variance.
    Uses SVD internally.
    """

    def __init__(self, n_components: int | None = None, variance_threshold: float = 0.95) -> None:
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.n_selected_: int = 0

    def fit(self, X: NDArray[np.float64]) -> "PCAReducer":
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        explained = (S ** 2) / (X.shape[0] - 1)
        total = explained.sum()
        self.explained_variance_ratio_ = explained / total if total > 0 else explained

        if self.n_components is not None:
            self.n_selected_ = min(self.n_components, len(S))
        else:
            cumulative = np.cumsum(self.explained_variance_ratio_)
            self.n_selected_ = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            self.n_selected_ = min(self.n_selected_, len(S))

        self.components_ = Vt[:self.n_selected_]
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: NDArray[np.float64]) -> NDArray[np.float64]:
        return X_reduced @ self.components_ + self.mean_
