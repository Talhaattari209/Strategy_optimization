"""Tests for the math engine modules."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from math_engine.linear_algebra import CovarianceMatrix, MatrixOps, OLSRegression, PCAReducer
from math_engine.markov_bayesian import DiscreteMarkovChain, BayesianUpdater, HiddenMarkovModel, MarketRegime
from math_engine.time_series import FourierCycleDetector, ARIMA, GARCH, RandomWalkBaseline
from math_engine.stochastic_processes import BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck, MonteCarloEngine
from math_engine.stochastic_calculus import ItoLemma, ItoIsometry, MartingaleChecker, GirsanovTransform, MicrostructureSDE
from math_engine.finance_models import BlackScholes, TechnicalFamaFrench, ExpectancyCalculator


class TestLinearAlgebra:
    def test_covariance_matrix(self):
        X = np.random.randn(100, 3)
        cov = CovarianceMatrix.compute(X)
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)

    def test_safe_inverse(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        inv = MatrixOps.safe_inverse(A)
        assert np.allclose(A @ inv, np.eye(2), atol=1e-10)

    def test_gram_schmidt(self):
        V = np.random.randn(3, 3)
        U = MatrixOps.gram_schmidt(V)
        for i in range(3):
            assert abs(np.linalg.norm(U[:, i]) - 1.0) < 1e-10
        assert abs(np.dot(U[:, 0], U[:, 1])) < 1e-10

    def test_ols_regression(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        true_beta = np.array([1.5, -0.5])
        y = X @ true_beta + 0.1 * np.random.randn(100)
        ols = OLSRegression().fit(X, y)
        assert abs(ols.beta[1] - 1.5) < 0.3
        assert abs(ols.beta[2] - (-0.5)) < 0.3

    def test_pca_reducer(self):
        X = np.random.randn(100, 10)
        pca = PCAReducer(n_components=3)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape == (100, 3)
        X_recon = pca.inverse_transform(X_reduced)
        assert X_recon.shape == (100, 10)

    def test_pca_variance_threshold(self):
        X = np.random.randn(100, 5)
        pca = PCAReducer(variance_threshold=0.9)
        pca.fit(X)
        assert pca.n_selected_ <= 5
        assert pca.n_selected_ >= 1


class TestMarkovBayesian:
    def test_markov_chain(self):
        states = np.array([0, 1, 1, 2, 0, 1, 2, 3, 0, 1])
        mc = DiscreteMarkovChain(4).fit(states)
        assert mc.transition_matrix.shape == (4, 4)
        assert np.allclose(mc.transition_matrix.sum(axis=1), 1.0, atol=1e-10)

    def test_bayesian_updater(self):
        bu = BayesianUpdater(4)
        likelihood = np.array([0.8, 0.1, 0.05, 0.05])
        posterior = bu.update(likelihood)
        assert np.argmax(posterior) == 0
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_hmm_fit_decode(self):
        np.random.seed(42)
        obs = np.concatenate([
            np.random.normal(1, 0.2, 50),
            np.random.normal(-1, 0.2, 50),
            np.random.normal(0, 0.5, 50),
        ])
        hmm = HiddenMarkovModel(n_states=3, n_iter=20)
        hmm.fit(obs)
        states = hmm.decode(obs)
        assert len(states) == len(obs)
        probs = hmm.predict_proba(obs)
        assert abs(probs.sum() - 1.0) < 0.1


class TestTimeSeries:
    def test_fourier(self):
        t = np.arange(200)
        signal = np.sin(2 * np.pi * t / 20) + 0.5 * np.sin(2 * np.pi * t / 50)
        fc = FourierCycleDetector(top_n=3)
        fc.fit(signal)
        periods = fc.get_dominant_periods()
        assert len(periods) == 3
        assert any(abs(p - 20) < 3 for p in periods)

    def test_arima_forecast(self):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.01)
        arima = ARIMA(p=3, d=1, q=1).fit(prices)
        forecast = arima.forecast(prices, steps=5)
        assert len(forecast) == 5
        assert all(np.isfinite(forecast))

    def test_garch(self):
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01
        garch = GARCH().fit(returns)
        vol = garch.current_vol(returns)
        assert vol > 0
        assert garch.alpha + garch.beta < 1.0

    def test_random_walk(self):
        forecast = RandomWalkBaseline.forecast(100.0, steps=5)
        assert all(f == 100.0 for f in forecast)


class TestStochasticProcesses:
    def test_brownian_motion(self):
        W = BrownianMotion.simulate(1.0, 100, n_paths=10, seed=42)
        assert W.shape == (10, 101)
        assert np.all(W[:, 0] == 0)

    def test_gbm(self):
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        paths = gbm.simulate(100, 1.0, 100, n_paths=1000, seed=42)
        assert paths.shape == (1000, 101)
        assert np.all(paths[:, 0] == 100)
        assert np.all(paths > 0)

    def test_ornstein_uhlenbeck(self):
        ou = OrnsteinUhlenbeck(theta=2.0, mu=50.0, sigma=1.0)
        paths = ou.simulate(45, 1.0, 100, n_paths=1000, seed=42)
        assert abs(np.mean(paths[:, -1]) - 50.0) < 5

    def test_monte_carlo(self):
        mc = MonteCarloEngine(n_paths=5000, seed=42)
        paths = mc.simulate_gbm(100, 0.05, 0.2, 1.0)
        stats = mc.path_statistics(paths)
        assert "mean_final" in stats
        assert "var_95" in stats
        assert stats["mean_final"] > 95


class TestStochasticCalculus:
    def test_ito_log_drift(self):
        drift = ItoLemma.log_price_drift(0.10, 0.20)
        assert abs(drift - 0.08) < 1e-10

    def test_ito_isometry(self):
        emp, theo = ItoIsometry.verify(lambda t: 1.0, T=1.0, n_steps=500, n_simulations=5000, seed=42)
        assert abs(emp - theo) < 0.2

    def test_martingale_checker(self):
        rng = np.random.default_rng(42)
        W = np.cumsum(rng.normal(0, 0.01, (1000, 50)), axis=1)
        is_mart, dev = MartingaleChecker.test(W, tolerance=0.1)
        assert isinstance(is_mart, (bool, np.bool_))

    def test_girsanov(self):
        g = GirsanovTransform(mu=0.10, r=0.02, sigma=0.20)
        assert abs(g.market_price_of_risk - 0.4) < 1e-10
        paths = g.risk_neutral_paths(100, 1.0, 50, n_paths=1000, seed=42)
        assert paths.shape == (1000, 51)

    def test_microstructure(self):
        ms = MicrostructureSDE()
        spread = ms.simulate_spread(0.0002, 1.0, 100, seed=42)
        assert len(spread) == 101
        assert np.all(spread > 0)
        cost = ms.expected_execution_cost(0.1, 0.0002, 1_000_000)
        assert cost > 0


class TestFinanceModels:
    def test_black_scholes_call(self):
        price = BlackScholes.call_price(100, 100, 1.0, 0.05, 0.20)
        assert 5 < price < 15

    def test_black_scholes_put(self):
        price = BlackScholes.put_price(100, 100, 1.0, 0.05, 0.20)
        assert 3 < price < 12

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        call = BlackScholes.call_price(S, K, T, r, sigma)
        put = BlackScholes.put_price(S, K, T, r, sigma)
        assert abs(call - put - S + K * np.exp(-r * T)) < 1e-10

    def test_greeks(self):
        delta = BlackScholes.delta(100, 100, 1.0, 0.05, 0.20)
        assert 0 < delta < 1
        gamma = BlackScholes.gamma(100, 100, 1.0, 0.05, 0.20)
        assert gamma > 0

    def test_fama_french(self):
        np.random.seed(42)
        factors = np.random.randn(100, 3)
        returns = factors @ np.array([0.5, -0.3, 0.2]) + 0.01 + 0.05 * np.random.randn(100)
        ff = TechnicalFamaFrench()
        ff.fit(returns, factors, ["momentum", "volume", "volatility"])
        attr = ff.attribution()
        assert "alpha" in attr
        assert "momentum" in attr

    def test_expectancy_calculator(self):
        calc = ExpectancyCalculator()
        result = calc.check_trade(1.1000, 1.1050, 1.0970, 0.10, 1/252)
        assert "reward_risk_ratio" in result
        assert result["reward_risk_ratio"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
