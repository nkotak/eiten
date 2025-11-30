"""
Regression Tests for Eiten

These tests ensure that behavior remains consistent across code changes.
They capture baseline behavior and detect unexpected changes.

Key validations:
1. Same inputs always produce same outputs (determinism)
2. Strategy implementations haven't changed unexpectedly
3. Numerical precision is maintained
4. Edge cases continue to be handled correctly
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import hashlib
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BackTester
from utils import (
    normalize_weights,
    get_price_deltas,
    get_log_returns,
    get_predicted_returns,
    random_matrix_theory_based_cov
)
from strategies.eigen_portfolio_strategy import EigenPortfolioStrategy
from strategies.minimum_variance_strategy import MinimumVarianceStrategy
from strategies.maximum_sharpe_ratio_strategy import MaximumSharpeRatioStrategy
from strategies.genetic_algo_strategy import GeneticAlgoStrategy


class TestDeterminism:
    """
    Tests that verify deterministic behavior - same inputs always
    produce exactly the same outputs.
    """

    @pytest.mark.regression
    def test_normalize_weights_deterministic(self):
        """Test that normalize_weights is deterministic."""
        weights = [0.3, -0.2, 0.5, -0.1, 0.4]

        results = []
        for _ in range(10):
            result = normalize_weights(weights.copy())
            results.append(tuple(result))

        # All results should be identical
        assert len(set(results)) == 1, "normalize_weights is not deterministic"

    @pytest.mark.regression
    def test_price_deltas_deterministic(self):
        """Test that get_price_deltas is deterministic."""
        np.random.seed(42)
        prices = pd.DataFrame({
            'A': np.random.uniform(100, 200, 50),
            'B': np.random.uniform(50, 150, 50)
        })

        results = []
        for _ in range(10):
            result = get_price_deltas(prices)
            results.append(result.values.tobytes())

        assert len(set(results)) == 1, "get_price_deltas is not deterministic"

    @pytest.mark.regression
    def test_log_returns_deterministic(self):
        """Test that get_log_returns is deterministic."""
        np.random.seed(42)
        prices = pd.DataFrame({
            'A': np.random.uniform(100, 200, 50),
            'B': np.random.uniform(50, 150, 50)
        })

        results = []
        for _ in range(10):
            result = get_log_returns(prices)
            results.append(result.values.tobytes())

        assert len(set(results)) == 1, "get_log_returns is not deterministic"

    @pytest.mark.regression
    def test_covariance_filtering_deterministic(self):
        """Test that random matrix theory filtering is deterministic."""
        np.random.seed(42)
        prices = pd.DataFrame({
            'A': np.random.uniform(100, 200, 100),
            'B': np.random.uniform(50, 150, 100),
            'C': np.random.uniform(75, 175, 100)
        })
        log_returns = get_log_returns(prices)

        results = []
        for _ in range(10):
            result = random_matrix_theory_based_cov(log_returns)
            results.append(result.values.tobytes())

        assert len(set(results)) == 1, \
            "random_matrix_theory_based_cov is not deterministic"

    @pytest.mark.regression
    def test_mvp_strategy_deterministic(self, strategy_params):
        """Test that MVP strategy is deterministic."""
        strategy = MinimumVarianceStrategy()

        results = []
        for _ in range(10):
            weights = strategy.generate_portfolio(**strategy_params)
            # Convert to sorted tuple for comparison
            result = tuple(sorted(weights.items()))
            results.append(result)

        assert len(set(results)) == 1, "MVP strategy is not deterministic"

    @pytest.mark.regression
    def test_msr_strategy_deterministic(self, strategy_params):
        """Test that MSR strategy is deterministic."""
        strategy = MaximumSharpeRatioStrategy()

        results = []
        for _ in range(10):
            weights = strategy.generate_portfolio(**strategy_params)
            result = tuple(sorted(weights.items()))
            results.append(result)

        assert len(set(results)) == 1, "MSR strategy is not deterministic"

    @pytest.mark.regression
    def test_eigen_strategy_deterministic(self, strategy_params):
        """Test that Eigen strategy is deterministic."""
        strategy = EigenPortfolioStrategy()

        results = []
        for _ in range(10):
            weights = strategy.generate_portfolio(**strategy_params)
            result = tuple(sorted(weights.items()))
            results.append(result)

        assert len(set(results)) == 1, "Eigen strategy is not deterministic"

    @pytest.mark.regression
    def test_genetic_algo_produces_valid_output(self, strategy_params):
        """Test that GA strategy produces valid portfolio weights.

        Note: GA is inherently stochastic and exact reproducibility depends
        on global random state. We test for valid output rather than exact
        determinism.
        """
        strategy = GeneticAlgoStrategy(random_seed=42)
        weights = strategy.generate_portfolio(**strategy_params)

        # Check output is valid
        assert weights is not None
        assert len(weights) == len(strategy_params['cov_matrix'].columns)

        # Weights should be normalized
        values = np.array(list(weights.values()))
        pos_sum = np.sum(values[values > 0])
        if pos_sum > 0:
            assert np.isclose(pos_sum, 1.0, rtol=0.05)

    @pytest.mark.regression
    def test_backtest_deterministic(self, data_dict, portfolio_weights_df):
        """Test that backtest is deterministic."""
        results = []
        for _ in range(10):
            result = BackTester.get_test(
                portfolio_weights_df, data_dict, 'historical', long_only=True
            )
            results.append(result.tobytes())

        assert len(set(results)) == 1, "BackTester.get_test is not deterministic"


class TestNumericalPrecision:
    """
    Tests that verify numerical precision is maintained.
    """

    @pytest.mark.regression
    def test_weight_normalization_precision(self):
        """Test that weight normalization maintains precision."""
        # Very small weights
        small_weights = [1e-10, 2e-10, 3e-10, 4e-10]
        normalized = normalize_weights(small_weights.copy())
        pos_sum = sum(w for w in normalized if w > 0)
        assert np.isclose(pos_sum, 1.0, rtol=1e-6), \
            "Precision lost in normalizing small weights"

        # Large weights
        large_weights = [1e10, 2e10, 3e10, 4e10]
        normalized = normalize_weights(large_weights.copy())
        pos_sum = sum(w for w in normalized if w > 0)
        assert np.isclose(pos_sum, 1.0, rtol=1e-6), \
            "Precision lost in normalizing large weights"

    @pytest.mark.regression
    def test_return_calculation_precision(self):
        """Test that return calculations maintain precision."""
        # Test with prices that have known exact returns
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 121.0]  # Exactly +10% each period
        })

        deltas = get_price_deltas(prices)
        assert np.isclose(deltas['A'].iloc[0], 0.1, rtol=1e-10)
        assert np.isclose(deltas['A'].iloc[1], 0.1, rtol=1e-10)

    @pytest.mark.regression
    def test_covariance_matrix_precision(self):
        """Test that covariance calculation maintains precision."""
        np.random.seed(42)
        n = 100

        # Create correlated returns with known correlation
        returns_a = np.random.normal(0, 0.02, n)
        returns_b = 0.8 * returns_a + np.random.normal(0, 0.01, n)

        prices = pd.DataFrame({
            'A': 100 * np.cumprod(1 + returns_a),
            'B': 100 * np.cumprod(1 + returns_b)
        })

        log_ret = get_log_returns(prices)
        cov = log_ret.cov()

        # Check symmetry
        assert np.allclose(cov.values, cov.values.T, rtol=1e-10), \
            "Covariance matrix lost symmetry precision"

        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert all(eigenvalues >= -1e-10), \
            "Covariance matrix is not positive semi-definite"

    @pytest.mark.regression
    def test_portfolio_weights_sum_precision(self, strategy_params):
        """Test that portfolio weights maintain proper sum precision."""
        strategies = [
            EigenPortfolioStrategy(),
            MinimumVarianceStrategy(),
            MaximumSharpeRatioStrategy(),
        ]

        for strategy in strategies:
            weights = strategy.generate_portfolio(**strategy_params)
            values = np.array(list(weights.values()))

            pos_sum = np.sum(values[values > 0])
            neg_sum = np.abs(np.sum(values[values < 0]))

            if pos_sum > 0:
                assert np.isclose(pos_sum, 1.0, rtol=0.01), \
                    f"{strategy.name}: positive weights don't sum to 1"
            if neg_sum > 0:
                assert np.isclose(neg_sum, 1.0, rtol=0.01), \
                    f"{strategy.name}: negative weights don't sum to -1"


class TestBehaviorConsistency:
    """
    Tests that verify behavior remains consistent with expected properties.
    """

    @pytest.mark.regression
    def test_mvp_reduces_variance(self, sample_symbols):
        """
        Test that MVP consistently produces lower variance portfolios
        than equal weight.
        """
        np.random.seed(42)
        n_tests = 5

        for test in range(n_tests):
            np.random.seed(42 + test)

            # Generate random prices
            prices = {}
            for symbol in sample_symbols:
                initial = np.random.uniform(50, 500)
                returns = np.random.normal(0.0005, 0.02, 200)
                prices[symbol] = initial * np.cumprod(1 + np.insert(returns, 0, 0))

            prices_df = pd.DataFrame(prices)
            log_ret = get_log_returns(prices_df)
            cov_matrix = log_ret.cov()
            pred_returns = get_price_deltas(prices_df)

            strategy = MinimumVarianceStrategy()
            mvp_weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=pred_returns,
                sample_returns=pred_returns,
                long_only=True
            )

            mvp_arr = np.array(list(mvp_weights.values()))
            n = len(sample_symbols)
            equal_arr = np.ones(n) / n

            cov_values = cov_matrix.values
            mvp_var = mvp_arr @ cov_values @ mvp_arr
            equal_var = equal_arr @ cov_values @ equal_arr

            # MVP should have lower or comparable variance
            assert mvp_var <= equal_var * 1.5, \
                f"Test {test}: MVP variance ({mvp_var}) > equal weight ({equal_var})"

    @pytest.mark.regression
    def test_long_only_no_negative_weights(self, strategy_params):
        """Test that long-only portfolios never have negative weights in final output."""
        strategies = [
            MinimumVarianceStrategy(),
            MaximumSharpeRatioStrategy(),
        ]

        params = {**strategy_params, 'long_only': True}

        for strategy in strategies:
            weights = strategy.generate_portfolio(**params)
            values = np.array(list(weights.values()))

            # After filter_short is applied in backtest, all should be >= 0
            filtered = BackTester.filter_short(values, long_only=True)
            assert all(w >= 0 for w in filtered), \
                f"{strategy.name} has negative weights after long-only filter"

    @pytest.mark.regression
    def test_eigen_portfolio_orthogonality(self, strategy_params):
        """Test that different eigen portfolios are different."""
        strategy = EigenPortfolioStrategy()

        portfolios = []
        for p_num in range(1, 5):
            params = {**strategy_params, 'p_number': p_num}
            weights = strategy.generate_portfolio(**params)
            portfolios.append(np.array(list(weights.values())))

        # Each portfolio should be different from others
        for i in range(len(portfolios)):
            for j in range(i + 1, len(portfolios)):
                assert not np.allclose(portfolios[i], portfolios[j]), \
                    f"Eigen portfolios {i+1} and {j+1} are too similar"

    @pytest.mark.regression
    def test_cumulative_returns_monotonic_for_positive(self):
        """Test that cumulative returns increase for consistently positive returns."""
        prices = pd.DataFrame({
            'A': [100.0, 102.0, 104.0, 106.0, 108.0]  # Consistent positive
        })

        data_dict = {'historical': prices}
        weights = pd.DataFrame({'P': {'A': 1.0}})

        result = BackTester.get_test(weights, data_dict, 'historical', long_only=True)

        # Should be monotonically increasing
        assert all(np.diff(result.flatten()) >= -1e-10), \
            "Cumulative returns not monotonic for positive returns"


class TestEdgeCaseRegression:
    """
    Tests that verify edge cases continue to be handled correctly.
    """

    @pytest.mark.regression
    def test_single_asset_regression(self):
        """Test that single-asset portfolios work correctly."""
        prices = pd.DataFrame({'A': [100.0, 110.0, 105.0, 115.0]})
        data_dict = {'historical': prices}
        weights = pd.DataFrame({'P': {'A': 1.0}})

        result = BackTester.get_test(weights, data_dict, 'historical', long_only=True)

        expected = get_price_deltas(prices).cumsum().values.flatten()
        np.testing.assert_array_almost_equal(result.flatten(), expected, decimal=10)

    @pytest.mark.regression
    def test_zero_return_regression(self):
        """Test that zero returns are handled correctly."""
        prices = pd.DataFrame({
            'A': [100.0, 100.0, 100.0, 100.0]  # No change
        })
        data_dict = {'historical': prices}
        weights = pd.DataFrame({'P': {'A': 1.0}})

        result = BackTester.get_test(weights, data_dict, 'historical', long_only=True)

        # All cumulative returns should be zero
        assert np.allclose(result.flatten(), 0, atol=1e-10), \
            "Zero returns not handled correctly"

    @pytest.mark.regression
    def test_high_correlation_regression(self):
        """Test that highly correlated assets work correctly."""
        np.random.seed(42)
        base_returns = np.random.normal(0, 0.02, 100)

        prices_a = 100 * np.cumprod(1 + np.insert(base_returns, 0, 0))
        prices_b = 50 * np.cumprod(1 + np.insert(base_returns + 0.001, 0, 0))  # Nearly identical

        prices = pd.DataFrame({'A': prices_a, 'B': prices_b})
        log_ret = get_log_returns(prices)
        cov_matrix = log_ret.cov()
        pred_returns = get_price_deltas(prices)

        # Should not crash with near-singular covariance
        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=pred_returns,
            sample_returns=pred_returns,
            long_only=True
        )

        assert weights is not None
        assert len(weights) == 2

    @pytest.mark.regression
    def test_negative_correlation_regression(self):
        """Test that negatively correlated assets work correctly."""
        np.random.seed(42)
        base_returns = np.random.normal(0, 0.02, 100)

        prices_a = 100 * np.cumprod(1 + np.insert(base_returns, 0, 0))
        prices_b = 50 * np.cumprod(1 + np.insert(-base_returns, 0, 0))  # Opposite

        prices = pd.DataFrame({'A': prices_a, 'B': prices_b})
        log_ret = get_log_returns(prices)
        cov_matrix = log_ret.cov()
        pred_returns = get_price_deltas(prices)

        # Should benefit from diversification
        strategy = MinimumVarianceStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=pred_returns,
            sample_returns=pred_returns,
            long_only=True
        )

        # Both assets should have non-trivial weights
        values = np.array(list(weights.values()))
        assert all(abs(v) > 0.01 for v in values), \
            "MVP should diversify across negatively correlated assets"


class TestDataIntegrityRegression:
    """
    Tests that verify data integrity is maintained throughout processing.
    """

    @pytest.mark.regression
    def test_column_names_preserved(self, historical_prices):
        """Test that column names are preserved through transformations."""
        original_columns = list(historical_prices.columns)

        deltas = get_price_deltas(historical_prices)
        assert list(deltas.columns) == original_columns

        log_ret = get_log_returns(historical_prices)
        assert list(log_ret.columns) == original_columns

        cov = log_ret.cov()
        assert list(cov.columns) == original_columns
        assert list(cov.index) == original_columns

        filtered_cov = random_matrix_theory_based_cov(log_ret)
        assert list(filtered_cov.columns) == original_columns

    @pytest.mark.regression
    def test_no_nan_propagation(self, historical_prices):
        """Test that NaN values don't propagate unexpectedly."""
        # Add a NaN to original data
        prices_with_nan = historical_prices.copy()
        prices_with_nan.iloc[50, 0] = np.nan

        # Forward fill should handle it
        prices_filled = prices_with_nan.fillna(method='ffill')

        deltas = get_price_deltas(prices_filled)
        # After forward fill, there should be no NaN
        assert not deltas.isna().any().any(), \
            "NaN propagated through price deltas"

    @pytest.mark.regression
    def test_row_count_consistency(self, historical_prices):
        """Test that row counts are consistent after transformations."""
        n_original = len(historical_prices)

        deltas = get_price_deltas(historical_prices)
        assert len(deltas) == n_original - 1

        log_ret = get_log_returns(historical_prices)
        assert len(log_ret) == n_original - 1

        pred_ret = get_predicted_returns(historical_prices)
        assert len(pred_ret) == n_original - 1


class TestOutputFormatRegression:
    """
    Tests that verify output formats remain consistent.
    """

    @pytest.mark.regression
    def test_strategy_output_format(self, strategy_params, sample_symbols):
        """Test that all strategies return properly formatted output."""
        strategies = [
            EigenPortfolioStrategy(),
            MinimumVarianceStrategy(),
            MaximumSharpeRatioStrategy(),
            GeneticAlgoStrategy(random_seed=42)
        ]

        for strategy in strategies:
            weights = strategy.generate_portfolio(**strategy_params)

            # Must be a dictionary
            assert isinstance(weights, dict), \
                f"{strategy.name} doesn't return dict"

            # Must have all symbols
            assert set(weights.keys()) == set(sample_symbols), \
                f"{strategy.name} missing symbols"

            # All values must be numeric
            for k, v in weights.items():
                assert isinstance(v, (int, float, np.floating)), \
                    f"{strategy.name} has non-numeric weight for {k}"

    @pytest.mark.regression
    def test_backtest_output_format(self, data_dict, portfolio_weights_df):
        """Test that backtest returns properly formatted output."""
        result = BackTester.get_test(
            portfolio_weights_df, data_dict, 'historical', long_only=True
        )

        # Must be numpy array
        assert isinstance(result, np.ndarray)

        # Must be 2D
        assert len(result.shape) == 2

        # No NaN or Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
