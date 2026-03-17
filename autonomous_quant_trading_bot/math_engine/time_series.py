"""
Phase 3 – Time Series Analysis
Fourier → Extract session-cycle components.
ARIMA/GARCH → Forecast volatility inside current session.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict


class FourierCycleDetector:
    """
    Fourier transform for cycle detection in price data.
    Extracts dominant session-cycle frequencies from returns.
    """

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n
        self.frequencies_: NDArray[np.float64] | None = None
        self.amplitudes_: NDArray[np.float64] | None = None
        self.phases_: NDArray[np.float64] | None = None

    def fit(self, series: NDArray[np.float64], sampling_period: float = 1.0) -> "FourierCycleDetector":
        n = len(series)
        fft_vals = np.fft.rfft(series - np.mean(series))
        freqs = np.fft.rfftfreq(n, d=sampling_period)
        amplitudes = np.abs(fft_vals) * 2.0 / n
        phases = np.angle(fft_vals)

        idx = np.argsort(amplitudes[1:])[::-1][:self.top_n] + 1
        self.frequencies_ = freqs[idx]
        self.amplitudes_ = amplitudes[idx]
        self.phases_ = phases[idx]
        return self

    def get_dominant_periods(self) -> NDArray[np.float64]:
        safe_freqs = np.where(self.frequencies_ > 1e-12, self.frequencies_, 1e-12)
        return 1.0 / safe_freqs

    def reconstruct(self, n_points: int) -> NDArray[np.float64]:
        t = np.arange(n_points, dtype=np.float64)
        result = np.zeros(n_points)
        for i in range(len(self.frequencies_)):
            result += self.amplitudes_[i] * np.cos(2 * np.pi * self.frequencies_[i] * t + self.phases_[i])
        return result

    def extract_features(self, series: NDArray[np.float64]) -> Dict[str, float]:
        self.fit(series)
        periods = self.get_dominant_periods()
        return {
            "dominant_period": float(periods[0]) if len(periods) > 0 else 0.0,
            "dominant_amplitude": float(self.amplitudes_[0]) if len(self.amplitudes_) > 0 else 0.0,
            "spectral_energy_ratio": float(np.sum(self.amplitudes_ ** 2) / max(np.var(series), 1e-12)),
        }


class ARIMA:
    """
    ARIMA(p,d,q) rolling forecaster.
    Simplified implementation for session-level forecasting.
    Uses OLS for AR coefficients and iterative residuals for MA.
    """

    def __init__(self, p: int = 5, d: int = 1, q: int = 1) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.ar_coeffs: NDArray[np.float64] | None = None
        self.ma_coeffs: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self._last_residuals: NDArray[np.float64] | None = None

    def _difference(self, series: NDArray[np.float64], order: int) -> NDArray[np.float64]:
        result = series.copy()
        for _ in range(order):
            result = np.diff(result)
        return result

    def fit(self, series: NDArray[np.float64]) -> "ARIMA":
        diff_series = self._difference(series, self.d)
        n = len(diff_series)
        if n <= self.p + 1:
            self.ar_coeffs = np.zeros(self.p)
            self.ma_coeffs = np.zeros(self.q)
            return self

        X = np.column_stack([diff_series[self.p - i - 1:n - i - 1] for i in range(self.p)])
        y = diff_series[self.p:]

        XtX = X.T @ X
        reg = np.eye(self.p) * 1e-6
        self.ar_coeffs = np.linalg.solve(XtX + reg, X.T @ y)
        self.intercept = np.mean(y) - np.mean(X, axis=0) @ self.ar_coeffs

        residuals = y - (X @ self.ar_coeffs + self.intercept)
        self._last_residuals = residuals

        if self.q > 0 and len(residuals) > self.q + 1:
            R = np.column_stack([residuals[self.q - i - 1:len(residuals) - i - 1] for i in range(self.q)])
            y_r = residuals[self.q:]
            RtR = R.T @ R + np.eye(self.q) * 1e-6
            self.ma_coeffs = np.linalg.solve(RtR, R.T @ y_r)
        else:
            self.ma_coeffs = np.zeros(self.q)

        return self

    def forecast(self, series: NDArray[np.float64], steps: int = 1) -> NDArray[np.float64]:
        diff_series = self._difference(series, self.d)
        history = list(diff_series[-self.p:])
        residuals = list(self._last_residuals[-self.q:]) if self._last_residuals is not None else [0.0] * self.q
        predictions = []

        for _ in range(steps):
            ar_part = sum(self.ar_coeffs[i] * history[-(i + 1)] for i in range(min(self.p, len(history))))
            ma_part = sum(self.ma_coeffs[i] * residuals[-(i + 1)] for i in range(min(self.q, len(residuals))))
            pred = self.intercept + ar_part + ma_part
            predictions.append(pred)
            history.append(pred)
            residuals.append(0.0)

        forecasts = np.array(predictions)
        last_val = series[-1]
        for _ in range(self.d):
            forecasts = np.cumsum(forecasts) + last_val
            last_val = forecasts[-1]

        return forecasts


class GARCH:
    """
    GARCH(1,1) volatility model.
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Used for volatility forecasting within current session.
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.long_run_var: float = 0.0

    def fit(self, returns: NDArray[np.float64], n_iter: int = 100) -> "GARCH":
        T = len(returns)
        if T < 10:
            self.omega = np.var(returns)
            return self

        var_r = np.var(returns)
        self.omega = var_r * 0.05
        self.alpha = 0.10
        self.beta = 0.85

        best_ll = -np.inf
        best_params = (self.omega, self.alpha, self.beta)

        for iteration in range(n_iter):
            sigma2 = np.full(T, var_r)
            for t in range(1, T):
                sigma2[t] = self.omega + self.alpha * returns[t - 1] ** 2 + self.beta * sigma2[t - 1]
                sigma2[t] = max(sigma2[t], 1e-12)

            ll = -0.5 * np.sum(np.log(sigma2) + returns ** 2 / sigma2)
            if ll > best_ll:
                best_ll = ll
                best_params = (self.omega, self.alpha, self.beta)

            grad_omega = 0.5 * np.sum(1.0 / sigma2 * (returns ** 2 / sigma2 - 1))
            lr = 1e-7 / (1 + iteration * 0.01)
            self.omega = max(1e-10, self.omega + lr * grad_omega)
            self.alpha = np.clip(self.alpha + lr * 0.01 * np.random.randn(), 0.01, 0.3)
            self.beta = np.clip(self.beta + lr * 0.01 * np.random.randn(), 0.5, 0.99)

            if self.alpha + self.beta >= 1.0:
                self.beta = 0.99 - self.alpha

        self.omega, self.alpha, self.beta = best_params
        denom = 1.0 - self.alpha - self.beta
        self.long_run_var = self.omega / denom if denom > 0.01 else var_r
        return self

    def forecast_variance(self, last_return: float, last_var: float, steps: int = 1) -> NDArray[np.float64]:
        variances = np.zeros(steps)
        prev_var = last_var
        prev_eps2 = last_return ** 2
        for t in range(steps):
            v = self.omega + self.alpha * prev_eps2 + self.beta * prev_var
            variances[t] = max(v, 1e-12)
            prev_eps2 = variances[t]
            prev_var = variances[t]
        return variances

    def current_vol(self, returns: NDArray[np.float64]) -> float:
        sigma2 = np.var(returns)
        for t in range(1, len(returns)):
            sigma2 = self.omega + self.alpha * returns[t - 1] ** 2 + self.beta * sigma2
        return np.sqrt(max(sigma2, 1e-12))


class RandomWalkBaseline:
    """Random walk baseline for comparison: E[X_{t+1}] = X_t."""

    @staticmethod
    def forecast(last_value: float, steps: int = 1) -> NDArray[np.float64]:
        return np.full(steps, last_value)

    @staticmethod
    def forecast_with_drift(series: NDArray[np.float64], steps: int = 1) -> NDArray[np.float64]:
        drift = np.mean(np.diff(series))
        return series[-1] + drift * np.arange(1, steps + 1)
