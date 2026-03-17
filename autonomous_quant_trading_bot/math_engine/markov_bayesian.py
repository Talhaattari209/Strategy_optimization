"""
Phase 2 – Probability & Bayesian
Regime detector uses session state + level proximity as hidden states.
Markov chains + Bayesian posterior updating + 4-state HMM.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from enum import IntEnum


class MarketRegime(IntEnum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOL = 3


class DiscreteMarkovChain:
    """Discrete Markov Chain with transition matrix estimation."""

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states
        self.transition_matrix = np.ones((n_states, n_states)) / n_states

    def fit(self, state_sequence: NDArray[np.int64]) -> "DiscreteMarkovChain":
        counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(state_sequence) - 1):
            s_from = state_sequence[i]
            s_to = state_sequence[i + 1]
            if 0 <= s_from < self.n_states and 0 <= s_to < self.n_states:
                counts[s_from, s_to] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition_matrix = counts / row_sums
        return self

    def predict_next(self, current_state: int, steps: int = 1) -> NDArray[np.float64]:
        state_vec = np.zeros(self.n_states)
        state_vec[current_state] = 1.0
        return state_vec @ np.linalg.matrix_power(self.transition_matrix, steps)

    def stationary_distribution(self) -> NDArray[np.float64]:
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        return stationary / stationary.sum()


class BayesianUpdater:
    """Bayesian posterior updating for regime probabilities."""

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states
        self.prior = np.ones(n_states) / n_states
        self.alpha = np.ones(n_states)  # Dirichlet concentration

    def update(self, likelihood: NDArray[np.float64]) -> NDArray[np.float64]:
        unnormalized = self.prior * likelihood
        total = unnormalized.sum()
        if total > 0:
            self.prior = unnormalized / total
        return self.prior.copy()

    def update_dirichlet(self, observed_state: int) -> NDArray[np.float64]:
        self.alpha[observed_state] += 1
        self.prior = self.alpha / self.alpha.sum()
        return self.prior.copy()

    def reset(self, prior: NDArray[np.float64] | None = None) -> None:
        if prior is not None:
            self.prior = prior.copy()
        else:
            self.prior = np.ones(self.n_states) / self.n_states
            self.alpha = np.ones(self.n_states)


class HiddenMarkovModel:
    """
    4-state HMM for market regime detection.
    States: TrendingUp, TrendingDown, Ranging, HighVol
    Uses Baum-Welch (EM) for fitting and Viterbi for decoding.
    Emission model: Gaussian per state.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 50) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.transition_matrix: NDArray[np.float64] = np.ones((n_states, n_states)) / n_states
        self.start_prob: NDArray[np.float64] = np.ones(n_states) / n_states
        self.means: NDArray[np.float64] = np.zeros(n_states)
        self.variances: NDArray[np.float64] = np.ones(n_states)

    def _gaussian_emission(self, x: float, state: int) -> float:
        var = max(self.variances[state], 1e-12)
        diff = x - self.means[state]
        return np.exp(-0.5 * diff ** 2 / var) / np.sqrt(2 * np.pi * var)

    def fit(self, observations: NDArray[np.float64]) -> "HiddenMarkovModel":
        T = len(observations)
        if T < 2:
            return self

        sorted_obs = np.sort(observations)
        quantiles = np.array_split(sorted_obs, self.n_states)
        self.means = np.array([q.mean() for q in quantiles])
        self.variances = np.array([max(q.var(), 1e-6) for q in quantiles])

        for _ in range(self.n_iter):
            # E-step: forward-backward
            alpha = np.zeros((T, self.n_states))
            beta = np.zeros((T, self.n_states))

            for s in range(self.n_states):
                alpha[0, s] = self.start_prob[s] * self._gaussian_emission(observations[0], s)
            alpha_sum = alpha[0].sum()
            if alpha_sum > 0:
                alpha[0] /= alpha_sum

            for t in range(1, T):
                for s in range(self.n_states):
                    alpha[t, s] = np.sum(alpha[t - 1] * self.transition_matrix[:, s]) * \
                                  self._gaussian_emission(observations[t], s)
                a_sum = alpha[t].sum()
                if a_sum > 0:
                    alpha[t] /= a_sum

            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                for s in range(self.n_states):
                    beta[t, s] = np.sum(
                        self.transition_matrix[s, :] *
                        np.array([self._gaussian_emission(observations[t + 1], j) for j in range(self.n_states)]) *
                        beta[t + 1]
                    )
                b_sum = beta[t].sum()
                if b_sum > 0:
                    beta[t] /= b_sum

            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1.0
            gamma /= gamma_sum

            # M-step
            self.start_prob = gamma[0]

            for i in range(self.n_states):
                for j in range(self.n_states):
                    numer = 0.0
                    denom = 0.0
                    for t in range(T - 1):
                        emit_j = self._gaussian_emission(observations[t + 1], j)
                        numer += alpha[t, i] * self.transition_matrix[i, j] * emit_j * beta[t + 1, j]
                        denom += gamma[t, i]
                    self.transition_matrix[i, j] = numer / denom if denom > 0 else 1.0 / self.n_states

            row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.transition_matrix /= row_sums

            for s in range(self.n_states):
                g_sum = gamma[:, s].sum()
                if g_sum > 0:
                    self.means[s] = np.sum(gamma[:, s] * observations) / g_sum
                    self.variances[s] = np.sum(gamma[:, s] * (observations - self.means[s]) ** 2) / g_sum
                    self.variances[s] = max(self.variances[s], 1e-6)

        return self

    def decode(self, observations: NDArray[np.float64]) -> NDArray[np.int64]:
        """Viterbi algorithm — most likely state sequence."""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=np.int64)

        for s in range(self.n_states):
            delta[0, s] = np.log(max(self.start_prob[s], 1e-300)) + \
                          np.log(max(self._gaussian_emission(observations[0], s), 1e-300))

        for t in range(1, T):
            for s in range(self.n_states):
                trans_probs = delta[t - 1] + np.log(np.maximum(self.transition_matrix[:, s], 1e-300))
                psi[t, s] = np.argmax(trans_probs)
                delta[t, s] = trans_probs[psi[t, s]] + \
                              np.log(max(self._gaussian_emission(observations[t], s), 1e-300))

        path = np.zeros(T, dtype=np.int64)
        path[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    def predict_proba(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return posterior state probabilities for the last observation."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            alpha[0, s] = self.start_prob[s] * self._gaussian_emission(observations[0], s)
        a_sum = alpha[0].sum()
        if a_sum > 0:
            alpha[0] /= a_sum

        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t, s] = np.sum(alpha[t - 1] * self.transition_matrix[:, s]) * \
                              self._gaussian_emission(observations[t], s)
            a_sum = alpha[t].sum()
            if a_sum > 0:
                alpha[t] /= a_sum

        return alpha[-1]
