"""
Phase 6 – Finance Models
Black-Scholes risk-neutral + technical Fama-French →
Expectancy check when price reacts at CHOCH/BOS.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from typing import Dict, Tuple

from .linear_algebra import OLSRegression


class BlackScholes:
    """
    Black-Scholes option pricing + risk-neutral valuation.
    Used for expectancy check at key structure levels.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0.0)
        _d1 = BlackScholes.d1(S, K, T, r, sigma)
        _d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * stats.norm.cdf(_d1) - K * np.exp(-r * T) * stats.norm.cdf(_d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0.0)
        _d1 = BlackScholes.d1(S, K, T, r, sigma)
        _d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * stats.norm.cdf(-_d2) - S * stats.norm.cdf(-_d1)

    @staticmethod
    def implied_vol(
        market_price: float, S: float, K: float, T: float, r: float,
        is_call: bool = True, tol: float = 1e-6, max_iter: int = 100
    ) -> float:
        sigma = 0.2
        for _ in range(max_iter):
            price = BlackScholes.call_price(S, K, T, r, sigma) if is_call else BlackScholes.put_price(S, K, T, r, sigma)
            vega = BlackScholes.vega(S, K, T, r, sigma)
            if abs(vega) < 1e-12:
                break
            sigma -= (price - market_price) / vega
            sigma = max(sigma, 1e-6)
            if abs(price - market_price) < tol:
                break
        return sigma

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
        _d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(stats.norm.cdf(_d1)) if is_call else float(stats.norm.cdf(_d1) - 1)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        _d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(stats.norm.pdf(_d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        _d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(S * stats.norm.pdf(_d1) * np.sqrt(T))

    @staticmethod
    def risk_neutral_expectancy(
        S: float, target: float, T: float, r: float, sigma: float
    ) -> float:
        """Expected profit under risk-neutral measure at a structure level."""
        if target > S:
            return BlackScholes.call_price(S, target, T, r, sigma)
        else:
            return BlackScholes.put_price(S, target, T, r, sigma)


class TechnicalFamaFrench:
    """
    Technical Fama-French style factor model.
    Uses custom factors: momentum, volume, volatility, session, level_proximity.
    R_i = α + β₁·momentum + β₂·volume + β₃·vol + β₄·session + β₅·level_prox + ε

    Provides expectancy attribution when price reacts at CHOCH/BOS.
    """

    def __init__(self) -> None:
        self.ols = OLSRegression()
        self.factor_names: list[str] = []
        self.factor_importance_: NDArray[np.float64] | None = None

    def fit(
        self, returns: NDArray[np.float64], factors: NDArray[np.float64], factor_names: list[str] | None = None
    ) -> "TechnicalFamaFrench":
        self.ols.fit(factors, returns)
        self.factor_names = factor_names or [f"factor_{i}" for i in range(factors.shape[1])]
        betas = self.ols.beta[1:]  # skip intercept
        factor_stds = np.std(factors, axis=0)
        self.factor_importance_ = np.abs(betas) * factor_stds
        total = self.factor_importance_.sum()
        if total > 0:
            self.factor_importance_ /= total
        return self

    def predict_return(self, factors: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.ols.predict(factors)

    @property
    def alpha(self) -> float:
        return float(self.ols.beta[0])

    @property
    def betas(self) -> NDArray[np.float64]:
        return self.ols.beta[1:]

    def attribution(self) -> Dict[str, float]:
        result = {"alpha": self.alpha}
        for i, name in enumerate(self.factor_names):
            result[name] = float(self.factor_importance_[i])
        return result

    def expectancy_at_level(
        self, momentum: float, volume: float, volatility: float, session_weight: float, level_proximity: float
    ) -> float:
        factors = np.array([[momentum, volume, volatility, session_weight, level_proximity]])
        return float(self.ols.predict(factors)[0])


class ExpectancyCalculator:
    """Combines BS risk-neutral check with Fama-French attribution at structure levels."""

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.r = risk_free_rate
        self.bs = BlackScholes()
        self.ff = TechnicalFamaFrench()

    def check_trade(
        self,
        entry: float,
        target: float,
        stop: float,
        sigma: float,
        T: float,
        factors: NDArray[np.float64] | None = None,
    ) -> Dict[str, float]:
        rn_value = BlackScholes.risk_neutral_expectancy(entry, target, T, self.r, sigma)
        rn_risk = BlackScholes.risk_neutral_expectancy(entry, stop, T, self.r, sigma)

        reward = abs(target - entry)
        risk = abs(entry - stop)
        rr_ratio = reward / risk if risk > 0 else 0.0

        result = {
            "risk_neutral_reward": rn_value,
            "risk_neutral_risk": rn_risk,
            "reward_risk_ratio": rr_ratio,
            "risk_neutral_edge": rn_value - rn_risk,
        }

        if factors is not None and len(factors.shape) == 2:
            ff_pred = float(self.ff.predict_return(factors)[0]) if self.ff.ols.beta is not None else 0.0
            result["ff_predicted_return"] = ff_pred

        return result
