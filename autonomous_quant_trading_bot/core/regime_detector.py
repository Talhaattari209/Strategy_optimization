"""
Regime Detector — uses HMM + GARCH + Fourier + session state + level proximity.
Markov + Bayesian regime detection with session and level context as hidden states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, Optional
from dataclasses import dataclass

from ..math_engine.markov_bayesian import HiddenMarkovModel, BayesianUpdater, MarketRegime, DiscreteMarkovChain
from ..math_engine.time_series import GARCH, FourierCycleDetector
from ..math_engine.linear_algebra import PCAReducer


@dataclass
class RegimeState:
    regime: MarketRegime
    probabilities: NDArray[np.float64]
    volatility: float
    dominant_cycle: float
    confidence: float
    details: Dict[str, float]


class RegimeDetector:
    """
    Combines HMM, GARCH, and Fourier for regime detection.
    Session state and level proximity are included as hidden state features.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.hmm = HiddenMarkovModel(n_states=4, n_iter=30)
        self.garch = GARCH()
        self.fourier = FourierCycleDetector(top_n=3)
        self.bayesian = BayesianUpdater(n_states=4)
        self.pca = PCAReducer(n_components=5)
        self.markov = DiscreteMarkovChain(n_states=4)

        self._fitted: bool = False
        self._last_regime: MarketRegime = MarketRegime.RANGING
        self._regime_history: list[int] = []

    def fit(self, returns: NDArray[np.float64]) -> "RegimeDetector":
        if len(returns) < 50:
            return self

        self.hmm.fit(returns)
        self.garch.fit(returns)
        self.fourier.fit(returns)

        states = self.hmm.decode(returns)
        self._regime_history = list(states)
        self.markov.fit(states)

        self._fitted = True
        return self

    def detect(
        self,
        returns: NDArray[np.float64],
        session_features: Dict[str, float] | None = None,
        level_features: Dict[str, float] | None = None,
    ) -> RegimeState:
        if not self._fitted or len(returns) < 10:
            return RegimeState(
                MarketRegime.RANGING,
                np.array([0.25, 0.25, 0.25, 0.25]),
                float(np.std(returns)) if len(returns) > 0 else 0.0,
                0.0, 0.0, {},
            )

        hmm_probs = self.hmm.predict_proba(returns)
        vol = self.garch.current_vol(returns)
        fourier_feats = self.fourier.extract_features(returns)

        likelihood = hmm_probs.copy()
        if vol > np.std(returns) * 1.5:
            likelihood[MarketRegime.HIGH_VOL] *= 1.5
        if session_features:
            if session_features.get("is_overlap", 0) > 0:
                likelihood[MarketRegime.HIGH_VOL] *= 1.2

        posterior = self.bayesian.update(likelihood)
        regime_idx = int(np.argmax(posterior))
        regime = MarketRegime(regime_idx)

        self._last_regime = regime
        self._regime_history.append(regime_idx)

        next_probs = self.markov.predict_next(regime_idx)

        return RegimeState(
            regime=regime,
            probabilities=posterior,
            volatility=vol,
            dominant_cycle=fourier_feats.get("dominant_period", 0.0),
            confidence=float(posterior[regime_idx]),
            details={
                "hmm_prob": float(hmm_probs[regime_idx]),
                "garch_vol": vol,
                "fourier_period": fourier_feats.get("dominant_period", 0),
                "next_trending_up": float(next_probs[MarketRegime.TRENDING_UP]),
                "next_trending_down": float(next_probs[MarketRegime.TRENDING_DOWN]),
                "next_ranging": float(next_probs[MarketRegime.RANGING]),
                "next_high_vol": float(next_probs[MarketRegime.HIGH_VOL]),
            },
        )

    def needs_refit(self, hours_since_last: float) -> bool:
        refit_hours = self.config.get("optimization", {}).get("regime_refit_hours", 4)
        return hours_since_last >= refit_hours

    @property
    def current_regime(self) -> MarketRegime:
        return self._last_regime
