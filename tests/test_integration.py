"""
Integration tests for Eiten.

These tests verify that all components work together correctly
in realistic scenarios. They test the full pipeline from
data to portfolio generation to backtesting.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BackTester
from utils import (
    get_price_deltas,
    get_log_returns,
    get_predicted_returns,
    random_matrix_theory_based_cov
)
from strategies.eigen_portfolio_strategy import EigenPortfolioStrategy
from strategies.minimum_variance_strategy import MinimumVarianceStrategy
from strategies.maximum_sharpe_ratio_strategy import MaximumSharpeRatioStrategy
from strategies.genetic_algo_strategy import GeneticAlgoStrategy


class TestFullPipeline:
    """
    Tests that verify the complete Eiten pipeline works correctly.
    """

    @pytest.mark.integration
    def test_complete_workflow_without_noise_filtering(
            self, synthetic_prices, future_bars, sample_symbols):
        """
        Test complete workflow without noise filtering.

        Pipeline:
        1. Split data into historical and future
        2. Calculate returns and covariance
        3. Generate portfolios with all strategies
        4. Backtest on historical data
        5. Forward test on future data
        """
        # Step 1: Split data
        historical = synthetic_prices.iloc[:-future_bars].copy()
        future = synthetic_prices.iloc[-future_bars:].copy()
        data_dict = {'historical': historical, 'future': future}

        # Step 2: Calculate returns and covariance
        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        # Step 3: Generate portfolios
        strategies = [
            EigenPortfolioStrategy(),
            MinimumVarianceStrategy(),
            MaximumSharpeRatioStrategy(),
            GeneticAlgoStrategy(random_seed=42)
        ]

        portfolios = {}
        for strategy in strategies:
            weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )
            portfolios[strategy.name] = weights

        # Convert to DataFrame
        portfolio_df = pd.DataFrame.from_dict(portfolios)

        # Step 4: Backtest
        backtest_results = BackTester.get_test(
            portfolio_df, data_dict, 'historical', long_only=True
        )

        # Step 5: Forward test
        forward_results = BackTester.get_test(
            portfolio_df, data_dict, 'future', long_only=True
        )

        # Validations
        assert backtest_results is not None
        assert forward_results is not None
        assert len(backtest_results) == len(historical) - 1
        assert len(forward_results) == len(future) - 1

        # Check all strategies produced valid weights
        for name, weights in portfolios.items():
            assert set(weights.keys()) == set(sample_symbols)

    @pytest.mark.integration
    def test_complete_workflow_with_noise_filtering(
            self, synthetic_prices, future_bars, sample_symbols):
        """
        Test complete workflow with random matrix theory noise filtering.
        """
        historical = synthetic_prices.iloc[:-future_bars].copy()
        future = synthetic_prices.iloc[-future_bars:].copy()
        data_dict = {'historical': historical, 'future': future}

        log_returns = get_log_returns(historical)

        # Apply noise filtering
        filtered_cov = random_matrix_theory_based_cov(log_returns)

        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        # Generate portfolios with filtered covariance
        strategies = [
            EigenPortfolioStrategy(),
            MinimumVarianceStrategy(),
            MaximumSharpeRatioStrategy(),
        ]

        portfolios = {}
        for strategy in strategies:
            weights = strategy.generate_portfolio(
                cov_matrix=filtered_cov,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )
            portfolios[strategy.name] = weights

        portfolio_df = pd.DataFrame.from_dict(portfolios)

        backtest_results = BackTester.get_test(
            portfolio_df, data_dict, 'historical', long_only=True
        )
        forward_results = BackTester.get_test(
            portfolio_df, data_dict, 'future', long_only=True
        )

        assert backtest_results is not None
        assert forward_results is not None

    @pytest.mark.integration
    def test_monte_carlo_simulation_workflow(
            self, synthetic_prices, future_bars, sample_symbols):
        """
        Test workflow including Monte Carlo simulation.
        """
        historical = synthetic_prices.iloc[:-future_bars].copy()
        future = synthetic_prices.iloc[-future_bars:].copy()
        data_dict = {'historical': historical, 'future': future}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        # Generate portfolio
        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=perc_returns,
            sample_returns=perc_returns,
            long_only=True
        )

        portfolio_df = pd.DataFrame({'MVP': weights})

        # Run Monte Carlo simulation
        np.random.seed(42)
        simulated_prices = BackTester.simulate_future_prices(
            data_dict, get_predicted_returns, simulation_timesteps=future_bars
        )

        data_dict['sim'] = simulated_prices

        # Backtest on simulated data
        sim_results = BackTester.get_test(
            portfolio_df, data_dict, 'sim', long_only=True
        )

        assert sim_results is not None
        assert not np.any(np.isnan(sim_results))

    @pytest.mark.integration
    def test_long_short_portfolio_workflow(
            self, synthetic_prices, future_bars, sample_symbols):
        """
        Test workflow with long-short portfolios (not long-only).
        """
        historical = synthetic_prices.iloc[:-future_bars].copy()
        future = synthetic_prices.iloc[-future_bars:].copy()
        data_dict = {'historical': historical, 'future': future}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        strategy = EigenPortfolioStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=perc_returns,
            sample_returns=perc_returns,
            long_only=False  # Allow shorts
        )

        portfolio_df = pd.DataFrame({'Eigen': weights})

        # Backtest with long_only=False
        results = BackTester.get_test(
            portfolio_df, data_dict, 'historical', long_only=False
        )

        assert results is not None


class TestDataFlowIntegrity:
    """
    Tests that verify data integrity throughout the pipeline.
    """

    @pytest.mark.integration
    def test_column_consistency(self, synthetic_prices, sample_symbols):
        """Test that column names are consistent throughout pipeline."""
        # Original columns
        original_cols = set(synthetic_prices.columns)

        # Through returns calculation
        price_deltas = get_price_deltas(synthetic_prices)
        assert set(price_deltas.columns) == original_cols

        log_returns = get_log_returns(synthetic_prices)
        assert set(log_returns.columns) == original_cols

        # Through covariance
        cov_matrix = log_returns.cov()
        assert set(cov_matrix.columns) == original_cols
        assert set(cov_matrix.index) == original_cols

        # Through filtered covariance
        filtered_cov = random_matrix_theory_based_cov(log_returns)
        assert set(filtered_cov.columns) == original_cols

        # Through strategy weights
        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=filtered_cov,
            p_number=2,
            pred_returns=price_deltas.mean(),
            perc_returns=price_deltas,
            sample_returns=price_deltas,
            long_only=True
        )
        assert set(weights.keys()) == original_cols

    @pytest.mark.integration
    def test_no_data_corruption(self, synthetic_prices, future_bars):
        """Test that data is not corrupted during processing."""
        original_prices = synthetic_prices.copy()

        # Process through pipeline
        historical = synthetic_prices.iloc[:-future_bars]
        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        filtered_cov = random_matrix_theory_based_cov(log_returns)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(synthetic_prices, original_prices)

    @pytest.mark.integration
    def test_temporal_alignment(self, synthetic_prices, future_bars):
        """Test that temporal alignment is maintained."""
        historical = synthetic_prices.iloc[:-future_bars]
        future = synthetic_prices.iloc[-future_bars:]

        # Returns should align properly
        hist_returns = get_price_deltas(historical)
        future_returns = get_price_deltas(future)

        # Last historical return index should be before first future return index
        assert hist_returns.index[-1] < future_returns.index[0]


class TestStrategyComparison:
    """
    Tests comparing strategy behaviors in integrated scenarios.
    """

    @pytest.mark.integration
    def test_strategy_diversity(self, synthetic_prices, future_bars, sample_symbols):
        """Test that different strategies produce meaningfully different results."""
        historical = synthetic_prices.iloc[:-future_bars]
        data_dict = {'historical': historical, 'future': synthetic_prices.iloc[-future_bars:]}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        strategies = {
            'Eigen': EigenPortfolioStrategy(),
            'MVP': MinimumVarianceStrategy(),
            'MSR': MaximumSharpeRatioStrategy(),
            'GA': GeneticAlgoStrategy(random_seed=42)
        }

        all_weights = {}
        for name, strategy in strategies.items():
            weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )
            all_weights[name] = np.array(list(weights.values()))

        # Check pairwise differences
        names = list(all_weights.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                w1, w2 = all_weights[names[i]], all_weights[names[j]]
                # Strategies should produce noticeably different weights
                diff = np.abs(w1 - w2).max()
                assert diff > 0.01, f"{names[i]} and {names[j]} are too similar"

    @pytest.mark.integration
    def test_all_strategies_complete_backtest(
            self, synthetic_prices, future_bars, sample_symbols):
        """Test that all strategies complete backtest without errors."""
        historical = synthetic_prices.iloc[:-future_bars]
        data_dict = {'historical': historical, 'future': synthetic_prices.iloc[-future_bars:]}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        strategies = [
            EigenPortfolioStrategy(),
            MinimumVarianceStrategy(),
            MaximumSharpeRatioStrategy(),
            GeneticAlgoStrategy(random_seed=42)
        ]

        for strategy in strategies:
            weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )

            portfolio_df = pd.DataFrame({strategy.name: weights})

            # Should complete without raising exceptions
            backtest = BackTester.get_test(
                portfolio_df, data_dict, 'historical', long_only=True
            )
            forward = BackTester.get_test(
                portfolio_df, data_dict, 'future', long_only=True
            )

            assert backtest is not None, f"{strategy.name} backtest failed"
            assert forward is not None, f"{strategy.name} forward test failed"


class TestEdgeCaseIntegration:
    """
    Integration tests for edge cases.
    """

    @pytest.mark.integration
    def test_minimal_data(self, sample_symbols):
        """Test with minimal amount of data."""
        np.random.seed(42)

        # Only 20 bars (minimum for meaningful analysis)
        prices = {}
        for symbol in sample_symbols:
            initial = np.random.uniform(50, 500)
            returns = np.random.normal(0.0005, 0.02, 20)
            prices[symbol] = initial * np.cumprod(1 + np.insert(returns, 0, 0))

        prices_df = pd.DataFrame(prices)
        historical = prices_df.iloc[:-5]
        future = prices_df.iloc[-5:]
        data_dict = {'historical': historical, 'future': future}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=perc_returns,
            sample_returns=perc_returns,
            long_only=True
        )

        portfolio_df = pd.DataFrame({'MVP': weights})

        # Should complete without errors
        results = BackTester.get_test(
            portfolio_df, data_dict, 'historical', long_only=True
        )

        assert results is not None

    @pytest.mark.integration
    def test_many_assets(self):
        """Test with large number of assets."""
        np.random.seed(42)
        n_assets = 30

        symbols = [f'STOCK_{i}' for i in range(n_assets)]
        prices = {}
        for symbol in symbols:
            initial = np.random.uniform(50, 500)
            returns = np.random.normal(0.0005, 0.02, 200)
            prices[symbol] = initial * np.cumprod(1 + np.insert(returns, 0, 0))

        prices_df = pd.DataFrame(prices)
        historical = prices_df.iloc[:-30]
        future = prices_df.iloc[-30:]
        data_dict = {'historical': historical, 'future': future}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        # Test with MVP (should handle large covariance matrix)
        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=perc_returns,
            sample_returns=perc_returns,
            long_only=True
        )

        assert len(weights) == n_assets

        portfolio_df = pd.DataFrame({'MVP': weights})
        results = BackTester.get_test(
            portfolio_df, data_dict, 'historical', long_only=True
        )

        assert results is not None

    @pytest.mark.integration
    def test_identical_prices(self):
        """Test handling of assets with identical prices."""
        np.random.seed(42)
        n_periods = 100

        # Two assets with nearly identical prices
        base_returns = np.random.normal(0.0005, 0.02, n_periods)
        prices = {
            'A': 100 * np.cumprod(1 + np.insert(base_returns, 0, 0)),
            'B': 100 * np.cumprod(1 + np.insert(base_returns, 0, 0)),  # Same as A
            'C': 50 * np.cumprod(1 + np.insert(np.random.normal(0.001, 0.015, n_periods), 0, 0))
        }

        prices_df = pd.DataFrame(prices)
        historical = prices_df.iloc[:-20]
        future = prices_df.iloc[-20:]
        data_dict = {'historical': historical, 'future': future}

        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        # Should handle singular/near-singular matrix
        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=perc_returns,
            sample_returns=perc_returns,
            long_only=True
        )

        # Should complete without crashing
        assert weights is not None
        assert len(weights) == 3
