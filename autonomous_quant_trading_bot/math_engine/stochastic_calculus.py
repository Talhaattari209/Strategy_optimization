"""
Phase 5 – Stochastic Calculus
Ito's Lemma + Girsanov → Optimal TP/SL and execution slicing at key levels.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple


class ItoLemma:
    """
    Ito's Lemma: if dX = a dt + b dW, and f(X,t) is C²,
    df = (∂f/∂t + a ∂f/∂x + ½ b² ∂²f/∂x²) dt + b ∂f/∂x dW

    Used for optimal TP/SL computation at key levels.
    """

    @staticmethod
    def apply(
        f: Callable[[float, float], float],
        df_dt: Callable[[float, float], float],
        df_dx: Callable[[float, float], float],
        d2f_dx2: Callable[[float, float], float],
        drift: float,
        diffusion: float,
        x: float,
        t: float,
        dt: float,
        dW: float,
    ) -> float:
        deterministic = (df_dt(x, t) + drift * df_dx(x, t) + 0.5 * diffusion ** 2 * d2f_dx2(x, t)) * dt
        stochastic = diffusion * df_dx(x, t) * dW
        return deterministic + stochastic

    @staticmethod
    def log_price_drift(mu: float, sigma: float) -> float:
        """For f(S)=ln(S) under GBM: drift of log-price = μ - σ²/2."""
        return mu - 0.5 * sigma ** 2


class ItoTaylorExpansion:
    """Ito-Taylor (Milstein) scheme for higher-order SDE discretization."""

    @staticmethod
    def milstein_step(
        x: float,
        drift: Callable[[float], float],
        diffusion: Callable[[float], float],
        d_diffusion: Callable[[float], float],
        dt: float,
        dW: float,
    ) -> float:
        a = drift(x)
        b = diffusion(x)
        b_prime = d_diffusion(x)
        return x + a * dt + b * dW + 0.5 * b * b_prime * (dW ** 2 - dt)

    @staticmethod
    def simulate(
        x0: float,
        drift: Callable[[float], float],
        diffusion: Callable[[float], float],
        d_diffusion: Callable[[float], float],
        T: float,
        n_steps: int,
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0

        for t in range(n_steps):
            dW = rng.normal(0, np.sqrt(dt), n_paths)
            for p in range(n_paths):
                paths[p, t + 1] = ItoTaylorExpansion.milstein_step(
                    paths[p, t], drift, diffusion, d_diffusion, dt, dW[p]
                )
        return paths


class ItoIsometry:
    """
    Ito Isometry: E[∫₀ᵀ f(t) dW(t)]² = E[∫₀ᵀ f(t)² dt]
    Used for variance computation of stochastic integrals.
    """

    @staticmethod
    def variance_of_stochastic_integral(
        f_squared_values: NDArray[np.float64], dt: float
    ) -> float:
        return float(np.sum(f_squared_values * dt))

    @staticmethod
    def verify(
        integrand: Callable[[float], float],
        T: float,
        n_steps: int = 1000,
        n_simulations: int = 5000,
        seed: int | None = None,
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)[:-1]

        integral_values = np.zeros(n_simulations)
        for sim in range(n_simulations):
            dW = rng.normal(0, np.sqrt(dt), n_steps)
            integral_values[sim] = np.sum(np.array([integrand(t) for t in t_grid]) * dW)

        empirical_var = np.var(integral_values)
        theoretical_var = np.sum(np.array([integrand(t) ** 2 for t in t_grid]) * dt)

        return float(empirical_var), float(theoretical_var)


class MartingaleChecker:
    """Check if a process satisfies the martingale property: E[X_{t+1}|F_t] = X_t."""

    @staticmethod
    def test(paths: NDArray[np.float64], tolerance: float = 0.05) -> Tuple[bool, float]:
        n_paths, n_steps = paths.shape
        max_deviation = 0.0
        for t in range(n_steps - 1):
            conditional_mean = np.mean(paths[:, t + 1])
            current_mean = np.mean(paths[:, t])
            deviation = abs(conditional_mean - current_mean) / max(abs(current_mean), 1e-12)
            max_deviation = max(max_deviation, deviation)
        return max_deviation < tolerance, max_deviation


class GirsanovTransform:
    """
    Girsanov measure change: risk-neutral pricing.
    Under P: dS = μS dt + σS dW^P
    Under Q: dS = rS dt + σS dW^Q, where dW^Q = dW^P + θ dt, θ = (μ-r)/σ

    Used for optimal entry execution and risk-neutral expectancy checks.
    """

    def __init__(self, mu: float, r: float, sigma: float) -> None:
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.market_price_of_risk = (mu - r) / sigma if sigma > 0 else 0.0

    def radon_nikodym(self, W_path: NDArray[np.float64], T: float) -> NDArray[np.float64]:
        theta = self.market_price_of_risk
        return np.exp(-theta * W_path[:, -1] - 0.5 * theta ** 2 * T)

    def risk_neutral_paths(
        self, S0: float, T: float, n_steps: int, n_paths: int = 10000, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Simulate paths under risk-neutral measure Q."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        Z = rng.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z[:, t]
            )
        return paths

    def optimal_entry_price(
        self, S0: float, target: float, T: float, n_paths: int = 10000
    ) -> float:
        paths = self.risk_neutral_paths(S0, T, 100, n_paths)
        if target > S0:
            hitting = np.min(paths, axis=1)
            return float(np.percentile(hitting, 25))
        else:
            hitting = np.max(paths, axis=1)
            return float(np.percentile(hitting, 75))


class MicrostructureSDE:
    """
    Microstructure SDE: bid-ask spread + market impact as OU process.
    dSpread = θ(μ_spread - Spread) dt + σ_spread dW
    Execution cost = f(spread, volume, urgency)
    """

    def __init__(
        self, theta: float = 5.0, mu_spread: float = 0.0002, sigma_spread: float = 0.0001
    ) -> None:
        self.theta = theta
        self.mu_spread = mu_spread
        self.sigma_spread = sigma_spread

    def simulate_spread(
        self, current_spread: float, T: float, n_steps: int = 100, seed: int | None = None
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        spread = np.zeros(n_steps + 1)
        spread[0] = current_spread

        for t in range(n_steps):
            dW = rng.normal(0, np.sqrt(dt))
            spread[t + 1] = (
                spread[t]
                + self.theta * (self.mu_spread - spread[t]) * dt
                + self.sigma_spread * dW
            )
            spread[t + 1] = max(spread[t + 1], 1e-6)
        return spread

    def expected_execution_cost(
        self, order_size: float, current_spread: float, daily_volume: float
    ) -> float:
        """Kyle's lambda-style market impact + spread cost."""
        spread_cost = current_spread / 2
        participation = order_size / max(daily_volume, 1)
        impact = self.sigma_spread * np.sqrt(participation)
        return spread_cost + impact

    def optimal_twap_slices(
        self, total_size: float, n_slices: int, current_spread: float, daily_volume: float
    ) -> NDArray[np.float64]:
        """Distribute order into slices minimizing total execution cost."""
        slice_size = total_size / n_slices
        costs = np.array([
            self.expected_execution_cost(slice_size, current_spread, daily_volume)
            for _ in range(n_slices)
        ])
        weights = 1.0 / (costs + 1e-12)
        weights /= weights.sum()
        return weights * total_size
