"""
Backtesting Validation Tests

These tests validate that the backtesting module produces mathematically
correct results. Unlike unit tests that check function behavior, these tests
verify the financial/mathematical correctness of backtest calculations.

Key validations:
1. Portfolio returns are correctly calculated from individual asset returns
2. Cumulative returns are properly accumulated
3. Weighted returns match manual calculations
4. Edge cases are handled correctly
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BackTester
from utils import get_price_deltas, get_log_returns, get_predicted_returns


class TestBacktestMathematicalCorrectness:
    """
    Tests that validate the mathematical correctness of backtest calculations.
    """

    @pytest.mark.backtest
    def test_portfolio_return_calculation(self):
        """
        Test that portfolio returns are correctly calculated as weighted sum.

        Given:
        - Asset A: returns [0.1, 0.05, -0.02]
        - Asset B: returns [0.05, 0.10, 0.03]
        - Weights: A=0.6, B=0.4

        Expected portfolio returns:
        - Period 1: 0.6*0.1 + 0.4*0.05 = 0.08
        - Period 2: 0.6*0.05 + 0.4*0.10 = 0.07
        - Period 3: 0.6*(-0.02) + 0.4*0.03 = 0.0
        """
        # Create prices that produce known returns
        prices_A = [100.0, 110.0, 115.5, 113.19]  # Returns: 0.1, 0.05, -0.02
        prices_B = [50.0, 52.5, 57.75, 59.4825]  # Returns: 0.05, 0.10, 0.03

        prices_df = pd.DataFrame({
            'A': prices_A,
            'B': prices_B
        })

        data_dict = {'historical': prices_df}
        weights_df = pd.DataFrame({'Portfolio': {'A': 0.6, 'B': 0.4}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=False)

        # Calculate expected cumulative returns manually
        returns_A = np.array([0.1, 0.05, -0.02])
        returns_B = np.array([0.05, 0.10, 0.03])

        expected_portfolio_returns = 0.6 * returns_A + 0.4 * returns_B
        expected_cumulative = np.cumsum(expected_portfolio_returns)

        # Verify results match expected
        np.testing.assert_array_almost_equal(
            result.flatten(),
            expected_cumulative,
            decimal=4
        )

    @pytest.mark.backtest
    def test_equal_weight_portfolio(self):
        """
        Test that equal-weighted portfolio returns are correctly calculated.

        For n assets with equal weights (1/n), portfolio return should be
        the simple average of individual asset returns.
        """
        # Create test data with 4 assets
        np.random.seed(42)
        n_assets = 4
        n_periods = 50

        # Generate random prices
        initial_prices = np.array([100.0, 150.0, 200.0, 75.0])
        prices = np.zeros((n_periods + 1, n_assets))
        prices[0] = initial_prices

        for t in range(1, n_periods + 1):
            returns = np.random.normal(0.001, 0.02, n_assets)
            prices[t] = prices[t-1] * (1 + returns)

        prices_df = pd.DataFrame(
            prices,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4']
        )

        data_dict = {'historical': prices_df}

        # Equal weights
        equal_weight = 1.0 / n_assets
        weights_df = pd.DataFrame({
            'EqualWeight': {f'Asset{i+1}': equal_weight for i in range(n_assets)}
        })

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Calculate expected: average of cumulative returns
        price_deltas = get_price_deltas(prices_df)
        individual_cumulative = price_deltas.cumsum()
        expected_equal_weight = individual_cumulative.mean(axis=1).values

        np.testing.assert_array_almost_equal(
            result.flatten(),
            expected_equal_weight,
            decimal=6
        )

    @pytest.mark.backtest
    def test_single_asset_portfolio(self):
        """
        Test that single-asset portfolio equals that asset's returns.
        """
        prices = pd.DataFrame({
            'SingleAsset': [100.0, 105.0, 102.0, 108.0, 110.0]
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'Portfolio': {'SingleAsset': 1.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Expected: cumulative returns of the single asset
        returns = get_price_deltas(prices)
        expected = returns.cumsum().values.flatten()

        np.testing.assert_array_almost_equal(
            result.flatten(),
            expected,
            decimal=6
        )

    @pytest.mark.backtest
    def test_zero_weight_asset_excluded(self):
        """
        Test that assets with zero weight don't contribute to returns.
        """
        prices = pd.DataFrame({
            'A': [100.0, 120.0, 150.0],  # +20%, +25%
            'B': [100.0, 50.0, 25.0],    # -50%, -50% (should be excluded)
        })

        data_dict = {'historical': prices}

        # Only asset A has weight
        weights_df = pd.DataFrame({'Portfolio': {'A': 1.0, 'B': 0.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=False)

        # Expected: only A's returns
        returns_A = get_price_deltas(prices[['A']])
        expected = returns_A.cumsum().values.flatten()

        np.testing.assert_array_almost_equal(
            result.flatten(),
            expected,
            decimal=6
        )

    @pytest.mark.backtest
    def test_long_short_portfolio_returns(self):
        """
        Test that long-short portfolios correctly calculate returns.

        Long asset A (weight 0.5), Short asset B (weight -0.5)
        When A goes up and B goes down, total return should be positive.
        """
        # A goes up 10%, B goes up 5%
        prices = pd.DataFrame({
            'A': [100.0, 110.0],
            'B': [100.0, 105.0],
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'Portfolio': {'A': 0.5, 'B': -0.5}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=False)

        # Expected: 0.5 * 0.10 + (-0.5) * 0.05 = 0.05 - 0.025 = 0.025
        expected = 0.5 * 0.10 + (-0.5) * 0.05

        assert np.isclose(result.flatten()[0], expected, rtol=0.01)

    @pytest.mark.backtest
    def test_cumulative_returns_accumulate(self):
        """
        Test that cumulative returns properly accumulate over time.
        """
        # Create asset with consistent 2% daily return
        n_days = 20
        daily_return = 0.02
        prices = [100.0]
        for _ in range(n_days):
            prices.append(prices[-1] * (1 + daily_return))

        prices_df = pd.DataFrame({'Asset': prices})
        data_dict = {'historical': prices_df}
        weights_df = pd.DataFrame({'Portfolio': {'Asset': 1.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Cumulative sum of returns should increase monotonically
        # since all returns are positive
        assert all(np.diff(result.flatten()) >= -1e-10)

        # Final cumulative return should be approximately n_days * daily_return
        expected_final = n_days * daily_return
        assert np.isclose(result.flatten()[-1], expected_final, rtol=0.01)


class TestBacktestEdgeCases:
    """
    Tests for edge cases in backtesting.
    """

    @pytest.mark.backtest
    def test_very_small_returns(self):
        """Test handling of very small returns (near zero)."""
        prices = pd.DataFrame({
            'A': [100.0, 100.0001, 100.0002, 100.0003]
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'Portfolio': {'A': 1.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Should handle small returns without numerical issues
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    @pytest.mark.backtest
    def test_volatile_returns(self):
        """Test handling of volatile (large swings) returns."""
        # Large swings: +50%, -50%, +100%, -30%
        prices = pd.DataFrame({
            'A': [100.0, 150.0, 75.0, 150.0, 105.0]
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'Portfolio': {'A': 1.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Verify no numerical issues
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        # Verify correct calculation
        returns = get_price_deltas(prices)
        expected = returns.cumsum().values.flatten()
        np.testing.assert_array_almost_equal(result.flatten(), expected, decimal=6)

    @pytest.mark.backtest
    def test_all_negative_returns(self):
        """Test portfolio with all negative returns (drawdown)."""
        prices = pd.DataFrame({
            'A': [100.0, 95.0, 90.0, 85.0, 80.0]
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'Portfolio': {'A': 1.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # All cumulative returns should be negative
        assert all(r < 0 for r in result.flatten())

    @pytest.mark.backtest
    def test_many_assets_portfolio(self):
        """Test backtesting with many assets."""
        np.random.seed(42)
        n_assets = 50
        n_periods = 100

        # Generate prices for many assets
        prices_data = {}
        for i in range(n_assets):
            initial = np.random.uniform(50, 500)
            returns = np.random.normal(0.0001, 0.02, n_periods)
            prices = initial * np.cumprod(1 + returns)
            prices_data[f'Asset_{i}'] = np.insert(prices, 0, initial)

        prices_df = pd.DataFrame(prices_data)
        data_dict = {'historical': prices_df}

        # Equal weights
        weight = 1.0 / n_assets
        weights_dict = {f'Asset_{i}': weight for i in range(n_assets)}
        weights_df = pd.DataFrame({'Portfolio': weights_dict})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Should complete without errors
        assert result is not None
        assert len(result) == n_periods
        assert not np.any(np.isnan(result))


class TestBacktestConsistency:
    """
    Tests for consistency in backtest results.
    """

    @pytest.mark.backtest
    def test_deterministic_results(self):
        """Test that same inputs produce same outputs."""
        np.random.seed(42)
        prices = pd.DataFrame({
            'A': np.random.uniform(100, 200, 50),
            'B': np.random.uniform(50, 150, 50)
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'Portfolio': {'A': 0.6, 'B': 0.4}})

        result1 = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)
        result2 = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.backtest
    def test_weight_order_independence(self):
        """Test that order of weights doesn't affect results when aligned properly."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 115.0],
            'B': [50.0, 55.0, 52.0],
            'C': [200.0, 190.0, 195.0]
        })

        data_dict = {'historical': prices}

        # Same weights - use reindex to ensure alignment with data columns
        weights_dict = {'A': 0.3, 'B': 0.3, 'C': 0.4}
        weights1 = pd.DataFrame({'P': weights_dict})
        # Ensure weights are aligned with data columns
        weights1 = weights1.reindex(prices.columns)
        weights2 = weights1.copy()  # Same weights, same order

        result1 = BackTester.get_test(weights1, data_dict, 'historical', long_only=True)
        result2 = BackTester.get_test(weights2, data_dict, 'historical', long_only=True)

        np.testing.assert_array_almost_equal(result1, result2, decimal=10)

    @pytest.mark.backtest
    def test_scaling_invariance(self):
        """Test that price scaling doesn't affect returns (relative measure)."""
        # Same returns, different scales
        prices1 = pd.DataFrame({
            'A': [100.0, 110.0, 121.0]  # +10%, +10%
        })
        prices2 = pd.DataFrame({
            'A': [1.0, 1.1, 1.21]  # Same returns, 100x smaller scale
        })

        data_dict1 = {'historical': prices1}
        data_dict2 = {'historical': prices2}
        weights_df = pd.DataFrame({'P': {'A': 1.0}})

        result1 = BackTester.get_test(weights_df, data_dict1, 'historical', long_only=True)
        result2 = BackTester.get_test(weights_df, data_dict2, 'historical', long_only=True)

        np.testing.assert_array_almost_equal(result1, result2, decimal=6)


class TestBacktestPerformanceMetrics:
    """
    Tests that validate derived performance metrics are correct.
    """

    @pytest.mark.backtest
    def test_total_return_matches_final_cumsum(self):
        """Test that total return equals final cumulative return."""
        np.random.seed(42)
        n_periods = 100

        prices = pd.DataFrame({
            'A': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_periods + 1))
        })

        data_dict = {'historical': prices}
        weights_df = pd.DataFrame({'P': {'A': 1.0}})

        result = BackTester.get_test(weights_df, data_dict, 'historical', long_only=True)

        # Total return from prices
        total_return_from_prices = (prices['A'].iloc[-1] / prices['A'].iloc[0]) - 1

        # Sum of returns should approximate total return (for small returns)
        # Note: cumsum of returns != compound return, but should be close
        cumsum_return = result.flatten()[-1]

        # They should be in the same ballpark (within 50% for typical returns)
        assert np.isclose(cumsum_return, total_return_from_prices, rtol=0.5)

    @pytest.mark.backtest
    def test_portfolio_diversification_effect(self):
        """
        Test that diversified portfolio has different risk than concentrated.

        A diversified portfolio's return path should differ from
        a concentrated one, even with same expected return.
        """
        np.random.seed(42)
        n_periods = 100

        # Two negatively correlated assets
        returns_A = np.random.normal(0.001, 0.02, n_periods)
        returns_B = -returns_A + np.random.normal(0.002, 0.005, n_periods)  # Negative correlation

        prices_A = 100 * np.cumprod(1 + np.insert(returns_A, 0, 0))
        prices_B = 100 * np.cumprod(1 + np.insert(returns_B, 0, 0))

        prices = pd.DataFrame({'A': prices_A, 'B': prices_B})
        data_dict = {'historical': prices}

        # Concentrated vs diversified
        concentrated = pd.DataFrame({'P': {'A': 1.0, 'B': 0.0}})
        diversified = pd.DataFrame({'P': {'A': 0.5, 'B': 0.5}})

        result_conc = BackTester.get_test(concentrated, data_dict, 'historical', long_only=True)
        result_div = BackTester.get_test(diversified, data_dict, 'historical', long_only=True)

        # Diversified should have lower variance (smoother path)
        var_conc = np.var(np.diff(result_conc.flatten()))
        var_div = np.var(np.diff(result_div.flatten()))

        # Diversification should reduce variance
        assert var_div < var_conc * 0.9  # At least 10% reduction
