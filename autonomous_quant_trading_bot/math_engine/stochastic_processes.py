"""
Phase 4 – Stochastic Processes
Monte Carlo + GBM + OU → Simulate paths from current level (HOD/LOD etc.)
until next session close.
GPU-accelerated via PyTorch when CUDA is available.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    _DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH_AVAILABLE = False
    _DEVICE = None


class BrownianMotion:
    """Standard Brownian Motion (Wiener Process) simulator."""

    @staticmethod
    def simulate(T: float, n_steps: int, n_paths: int = 1, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        W = np.cumsum(dW, axis=1)
        W = np.hstack([np.zeros((n_paths, 1)), W])
        return W


class GeometricBrownianMotion:
    """
    GBM: dS = μS dt + σS dW
    Exact solution: S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.01) -> None:
        self.mu = mu
        self.sigma = sigma

    def simulate(
        self, S0: float, T: float, n_steps: int, n_paths: int = 10000, seed: int | None = None
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        Z = rng.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z[:, t]
            )
        return paths

    def expected_value(self, S0: float, T: float) -> float:
        return S0 * np.exp(self.mu * T)

    def variance(self, S0: float, T: float) -> float:
        return S0 ** 2 * np.exp(2 * self.mu * T) * (np.exp(self.sigma ** 2 * T) - 1)


class OrnsteinUhlenbeck:
    """
    OU Process: dX = θ(μ - X) dt + σ dW
    Mean-reverting process for spread modeling and dynamic trailing.
    """

    def __init__(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 0.01) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def fit(self, series: NDArray[np.float64], dt: float = 1.0) -> "OrnsteinUhlenbeck":
        n = len(series) - 1
        if n < 2:
            return self

        X = series[:-1]
        Y = series[1:]
        dX = Y - X

        X_mean = np.mean(X)
        dX_mean = np.mean(dX)

        cov_XdX = np.mean((X - X_mean) * (dX - dX_mean))
        var_X = np.var(X)

        if var_X > 1e-12:
            self.theta = max(-cov_XdX / (var_X * dt), 1e-6)
            self.mu = np.mean(series)
            residuals = dX - self.theta * (self.mu - X) * dt
            self.sigma = max(np.std(residuals) / np.sqrt(dt), 1e-6)

        return self

    def simulate(
        self, X0: float, T: float, n_steps: int, n_paths: int = 10000, seed: int | None = None
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = X0

        Z = rng.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            paths[:, t + 1] = (
                paths[:, t]
                + self.theta * (self.mu - paths[:, t]) * dt
                + self.sigma * np.sqrt(dt) * Z[:, t]
            )
        return paths

    def half_life(self) -> float:
        return np.log(2) / self.theta if self.theta > 0 else np.inf

    def stationary_variance(self) -> float:
        return self.sigma ** 2 / (2 * self.theta) if self.theta > 0 else np.inf


class MonteCarloEngine:
    """
    Monte Carlo path generator.
    Runs 10,000 paths for robust probability estimation.
    Uses GPU (CUDA) via PyTorch when available, falls back to NumPy on CPU.
    """

    def __init__(self, n_paths: int = 10000, seed: int | None = None) -> None:
        self.n_paths = n_paths
        self.seed = seed

    def simulate_gbm(
        self, S0: float, mu: float, sigma: float, T: float, n_steps: int = 100
    ) -> NDArray[np.float64]:
        if _TORCH_AVAILABLE:
            return self._simulate_gbm_gpu(S0, mu, sigma, T, n_steps)
        gbm = GeometricBrownianMotion(mu, sigma)
        return gbm.simulate(S0, T, n_steps, self.n_paths, self.seed)

    def _simulate_gbm_gpu(
        self, S0: float, mu: float, sigma: float, T: float, n_steps: int
    ) -> NDArray[np.float64]:
        dt = T / n_steps
        if self.seed is not None:
            _torch.manual_seed(self.seed)
        Z = _torch.randn(self.n_paths, n_steps, device=_DEVICE)
        log_ret = (_torch.tensor((mu - 0.5 * sigma ** 2) * dt, device=_DEVICE)
                   + _torch.tensor(sigma * float(np.sqrt(dt)), device=_DEVICE) * Z)
        log_paths = _torch.cat([
            _torch.zeros(self.n_paths, 1, device=_DEVICE),
            _torch.cumsum(log_ret, dim=1)
        ], dim=1)
        paths = S0 * _torch.exp(log_paths)
        return paths.cpu().numpy().astype(np.float64)

    def simulate_ou(
        self, X0: float, theta: float, mu: float, sigma: float, T: float, n_steps: int = 100
    ) -> NDArray[np.float64]:
        if _TORCH_AVAILABLE:
            return self._simulate_ou_gpu(X0, theta, mu, sigma, T, n_steps)
        ou = OrnsteinUhlenbeck(theta, mu, sigma)
        return ou.simulate(X0, T, n_steps, self.n_paths, self.seed)

    def _simulate_ou_gpu(
        self, X0: float, theta: float, mu: float, sigma: float, T: float, n_steps: int
    ) -> NDArray[np.float64]:
        dt = T / n_steps
        if self.seed is not None:
            _torch.manual_seed(self.seed)
        Z = _torch.randn(self.n_paths, n_steps, device=_DEVICE)
        paths = _torch.zeros(self.n_paths, n_steps + 1, device=_DEVICE)
        paths[:, 0] = X0
        _theta = _torch.tensor(theta * dt, device=_DEVICE)
        _mu    = _torch.tensor(mu, device=_DEVICE)
        _sig   = _torch.tensor(sigma * float(np.sqrt(dt)), device=_DEVICE)
        for t in range(n_steps):
            paths[:, t + 1] = (paths[:, t]
                               + _theta * (_mu - paths[:, t])
                               + _sig * Z[:, t])
        return paths.cpu().numpy().astype(np.float64)

    def probability_above(self, paths: NDArray[np.float64], level: float) -> float:
        return float(np.mean(paths[:, -1] > level))

    def probability_below(self, paths: NDArray[np.float64], level: float) -> float:
        return float(np.mean(paths[:, -1] < level))

    def expected_max(self, paths: NDArray[np.float64]) -> float:
        return float(np.mean(np.max(paths, axis=1)))

    def expected_min(self, paths: NDArray[np.float64]) -> float:
        return float(np.mean(np.min(paths, axis=1)))

    def var_at_confidence(self, paths: NDArray[np.float64], confidence: float = 0.95) -> float:
        final_returns = (paths[:, -1] - paths[:, 0]) / paths[:, 0]
        return float(np.percentile(final_returns, (1 - confidence) * 100))

    def path_statistics(self, paths: NDArray[np.float64]) -> Dict[str, float]:
        finals = paths[:, -1]
        return {
            "mean_final": float(np.mean(finals)),
            "std_final": float(np.std(finals)),
            "median_final": float(np.median(finals)),
            "prob_positive_return": float(np.mean(finals > paths[:, 0])),
            "expected_max": self.expected_max(paths),
            "expected_min": self.expected_min(paths),
            "var_95": self.var_at_confidence(paths, 0.95),
        }
