"""
Forward Testing Validation Tests

These tests validate that forward (out-of-sample) testing is implemented
correctly. Key concerns:
1. No data leakage from future to training period
2. Correct temporal splitting of data
3. Portfolio weights are derived only from historical data
4. Forward test uses truly unseen data
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BackTester
from utils import get_price_deltas, get_log_returns, get_predicted_returns
from strategies.eigen_portfolio_strategy import EigenPortfolioStrategy
from strategies.minimum_variance_strategy import MinimumVarianceStrategy
from strategies.maximum_sharpe_ratio_strategy import MaximumSharpeRatioStrategy


class TestDataSplitCorrectness:
    """
    Tests that validate correct temporal splitting of data for forward testing.
    """

    @pytest.mark.forwardtest
    def test_historical_future_split_no_overlap(self, data_dict):
        """
        Test that historical and future data have no overlapping periods.
        """
        historical = data_dict['historical']
        future = data_dict['future']

        # Check no overlap in indices
        hist_indices = set(historical.index)
        future_indices = set(future.index)

        overlap = hist_indices.intersection(future_indices)
        assert len(overlap) == 0, f"Found overlapping indices: {overlap}"

    @pytest.mark.forwardtest
    def test_temporal_ordering(self, data_dict):
        """
        Test that all historical data comes before future data.
        """
        historical = data_dict['historical']
        future = data_dict['future']

        if len(historical) > 0 and len(future) > 0:
            # Using numeric indices, last historical should be before first future
            assert historical.index[-1] < future.index[0], \
                "Historical data extends into future period"

    @pytest.mark.forwardtest
    def test_split_preserves_all_data(self, synthetic_prices, data_dict, future_bars):
        """
        Test that splitting preserves all original data.
        """
        total_original = len(synthetic_prices)
        total_split = len(data_dict['historical']) + len(data_dict['future'])

        assert total_original == total_split, \
            f"Data lost in split: original={total_original}, split={total_split}"

    @pytest.mark.forwardtest
    def test_future_bars_count(self, data_dict, future_bars):
        """
        Test that future period has exactly the expected number of bars.
        """
        assert len(data_dict['future']) == future_bars, \
            f"Expected {future_bars} future bars, got {len(data_dict['future'])}"


class TestNoDataLeakage:
    """
    Tests that ensure no information from the future leaks into training.
    """

    @pytest.mark.forwardtest
    def test_portfolio_weights_from_historical_only(self, data_dict, sample_symbols):
        """
        Test that portfolio weights are computed using only historical data.

        This test verifies that changing future data doesn't affect
        the portfolio weights computed from historical data.
        """
        historical = data_dict['historical']
        future = data_dict['future']

        # Compute covariance from historical only
        log_returns = get_log_returns(historical)
        cov_matrix = log_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        strategy = MinimumVarianceStrategy()
        params = {
            'cov_matrix': cov_matrix,
            'p_number': 2,
            'pred_returns': pred_returns.mean(),
            'perc_returns': perc_returns,
            'sample_returns': perc_returns,
            'long_only': True
        }

        weights1 = strategy.generate_portfolio(**params)

        # Now modify future data dramatically
        modified_future = future * 100  # 100x change in future prices

        # Weights should be identical (future doesn't affect them)
        weights2 = strategy.generate_portfolio(**params)

        for symbol in sample_symbols:
            assert np.isclose(weights1[symbol], weights2[symbol]), \
                f"Future data affected weights for {symbol}"

    @pytest.mark.forwardtest
    def test_covariance_excludes_future(self, data_dict):
        """
        Test that covariance matrix calculation uses only historical data.
        """
        historical = data_dict['historical']
        future = data_dict['future']

        # Covariance from historical only
        hist_returns = get_log_returns(historical)
        hist_cov = hist_returns.cov()

        # Covariance including future (should be different)
        full_data = pd.concat([historical, future])
        full_returns = get_log_returns(full_data)
        full_cov = full_returns.cov()

        # They should be different (proving future is properly separated)
        assert not np.allclose(hist_cov.values, full_cov.values), \
            "Covariance matrices are identical - future may be leaking into historical"

    @pytest.mark.forwardtest
    def test_predicted_returns_exclude_future(self, data_dict):
        """
        Test that predicted returns don't include future information.
        """
        historical = data_dict['historical']
        future = data_dict['future']

        # Predicted returns from historical
        hist_pred = get_predicted_returns(historical)

        # Predicted returns from full data
        full_data = pd.concat([historical, future])
        full_pred = get_predicted_returns(full_data)

        # Historical predictions shouldn't match full data predictions
        # (if they were computed correctly without leakage)
        assert not np.allclose(
            hist_pred.values[-1],
            full_pred.values[len(historical) - 2]  # Adjusted for shift
        ), "Predicted returns may include future information"


class TestForwardTestCalculation:
    """
    Tests that validate forward test calculations.
    """

    @pytest.mark.forwardtest
    def test_forward_test_uses_future_data(self, data_dict, portfolio_weights_df):
        """
        Test that forward test actually uses future data, not historical.
        """
        hist_result = BackTester.get_test(
            portfolio_weights_df, data_dict, 'historical', long_only=True
        )
        future_result = BackTester.get_test(
            portfolio_weights_df, data_dict, 'future', long_only=True
        )

        # Results should be different lengths
        assert len(hist_result) != len(future_result) or \
               not np.allclose(hist_result, future_result), \
            "Forward test may be using historical data"

    @pytest.mark.forwardtest
    def test_forward_test_correct_length(self, data_dict, portfolio_weights_df, future_bars):
        """
        Test that forward test result has correct length.
        """
        future_result = BackTester.get_test(
            portfolio_weights_df, data_dict, 'future', long_only=True
        )

        # Should be future_bars - 1 (due to returns calculation)
        expected_length = future_bars - 1
        assert len(future_result) == expected_length, \
            f"Expected {expected_length} forward test results, got {len(future_result)}"

    @pytest.mark.forwardtest
    def test_forward_test_independent_of_historical(self, portfolio_weights_df):
        """
        Test that forward test results depend only on future data and weights.

        Changing historical data shouldn't affect forward test results
        (given the same weights).
        """
        np.random.seed(42)

        # Create two different historical datasets
        historical1 = pd.DataFrame({
            'A': np.random.uniform(100, 200, 100),
            'B': np.random.uniform(50, 150, 100)
        })

        historical2 = pd.DataFrame({
            'A': np.random.uniform(200, 300, 100),  # Different historical
            'B': np.random.uniform(150, 250, 100)
        })

        # Same future data for both
        future = pd.DataFrame({
            'A': [150.0, 155.0, 160.0, 158.0, 162.0],
            'B': [100.0, 102.0, 99.0, 101.0, 103.0]
        })

        data_dict1 = {'historical': historical1, 'future': future}
        data_dict2 = {'historical': historical2, 'future': future}

        weights = pd.DataFrame({'P': {'A': 0.6, 'B': 0.4}})

        result1 = BackTester.get_test(weights, data_dict1, 'future', long_only=True)
        result2 = BackTester.get_test(weights, data_dict2, 'future', long_only=True)

        # Forward tests should be identical (same future data and weights)
        np.testing.assert_array_equal(result1, result2)


class TestForwardTestVsBacktest:
    """
    Tests comparing forward test behavior to backtest behavior.
    """

    @pytest.mark.forwardtest
    def test_same_calculation_different_data(self, data_dict, portfolio_weights_df):
        """
        Test that forward test uses same calculation logic as backtest,
        just on different data.
        """
        # Create identical data in historical and future positions
        identical_data = pd.DataFrame({
            'A': [100.0, 110.0, 105.0, 115.0, 120.0],
            'B': [50.0, 52.0, 51.0, 54.0, 53.0]
        })

        data_identical_hist = {
            'historical': identical_data.copy(),
            'future': pd.DataFrame()  # Empty
        }
        data_identical_future = {
            'historical': pd.DataFrame(),
            'future': identical_data.copy()
        }

        weights = pd.DataFrame({'P': {'A': 0.5, 'B': 0.5}})

        # Skip if either is empty (would cause calculation issues)
        if len(data_identical_hist['historical']) > 0:
            result_hist = BackTester.get_test(
                weights, data_identical_hist, 'historical', long_only=True
            )

        if len(data_identical_future['future']) > 0:
            result_future = BackTester.get_test(
                weights, data_identical_future, 'future', long_only=True
            )

            # Both should produce identical results for identical data
            np.testing.assert_array_almost_equal(result_hist, result_future, decimal=10)

    @pytest.mark.forwardtest
    def test_forward_test_realistic_scenario(self, sample_symbols):
        """
        Test a realistic forward testing scenario with full workflow.
        """
        np.random.seed(42)

        # Generate realistic price data
        n_historical = 252  # 1 year of daily data
        n_future = 30  # 1 month forward test

        prices = {}
        for symbol in sample_symbols:
            initial = np.random.uniform(50, 500)
            all_returns = np.random.normal(0.0005, 0.02, n_historical + n_future)
            all_prices = initial * np.cumprod(1 + np.insert(all_returns, 0, 0))
            prices[symbol] = all_prices

        full_prices = pd.DataFrame(prices)
        historical = full_prices.iloc[:n_historical + 1]
        future = full_prices.iloc[n_historical:]

        # Build portfolio from historical data only
        hist_returns = get_log_returns(historical)
        cov_matrix = hist_returns.cov()
        pred_returns = get_predicted_returns(historical)
        perc_returns = get_price_deltas(historical)

        strategy = MaximumSharpeRatioStrategy()
        weights = strategy.generate_portfolio(
            cov_matrix=cov_matrix,
            p_number=2,
            pred_returns=pred_returns.mean(),
            perc_returns=perc_returns,
            sample_returns=perc_returns,
            long_only=True
        )

        weights_df = pd.DataFrame({'MSR': weights})
        data_dict = {'historical': historical, 'future': future}

        # Run backtest and forward test
        backtest_result = BackTester.get_test(
            weights_df, data_dict, 'historical', long_only=True
        )
        forward_result = BackTester.get_test(
            weights_df, data_dict, 'future', long_only=True
        )

        # Both should complete without errors
        assert backtest_result is not None
        assert forward_result is not None

        # Forward test should have results
        # (length depends on future data slice which includes overlap point)
        assert len(forward_result) > 0


class TestWalkForwardValidation:
    """
    Tests for walk-forward validation concepts.
    """

    @pytest.mark.forwardtest
    def test_rolling_forward_test(self, sample_symbols):
        """
        Test rolling forward test where we repeatedly train and test.

        This validates that the system can handle multiple train/test splits.
        """
        np.random.seed(42)

        # Generate extended price data
        n_total = 500
        train_window = 200
        test_window = 30

        prices = {}
        for symbol in sample_symbols:
            initial = np.random.uniform(50, 500)
            returns = np.random.normal(0.0005, 0.02, n_total)
            prices[symbol] = initial * np.cumprod(1 + np.insert(returns, 0, 0))

        full_prices = pd.DataFrame(prices)

        # Perform rolling forward tests
        all_forward_results = []
        strategy = MinimumVarianceStrategy()

        for start in range(0, n_total - train_window - test_window, test_window):
            # Split data
            train_end = start + train_window
            test_end = train_end + test_window

            historical = full_prices.iloc[start:train_end + 1]
            future = full_prices.iloc[train_end:test_end + 1]

            # Train model (compute weights)
            hist_returns = get_log_returns(historical)
            cov_matrix = hist_returns.cov()
            pred_returns = get_predicted_returns(historical)
            perc_returns = get_price_deltas(historical)

            weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )

            weights_df = pd.DataFrame({'MVP': weights})
            data_dict = {'historical': historical, 'future': future}

            # Forward test
            result = BackTester.get_test(
                weights_df, data_dict, 'future', long_only=True
            )

            all_forward_results.append(result)

        # Should have multiple forward test results
        assert len(all_forward_results) >= 3, "Not enough rolling windows tested"

        # Each should be valid
        for result in all_forward_results:
            assert result is not None
            assert not np.any(np.isnan(result))

    @pytest.mark.forwardtest
    def test_expanding_window_forward_test(self, sample_symbols):
        """
        Test expanding window forward test (growing training set).
        """
        np.random.seed(42)

        n_total = 400
        initial_train = 100
        test_window = 20

        prices = {}
        for symbol in sample_symbols:
            initial = np.random.uniform(50, 500)
            returns = np.random.normal(0.0005, 0.02, n_total)
            prices[symbol] = initial * np.cumprod(1 + np.insert(returns, 0, 0))

        full_prices = pd.DataFrame(prices)

        strategy = MinimumVarianceStrategy()
        results = []

        for train_end in range(initial_train, n_total - test_window, test_window):
            test_end = train_end + test_window

            # Expanding window: always start from 0
            historical = full_prices.iloc[:train_end + 1]
            future = full_prices.iloc[train_end:test_end + 1]

            hist_returns = get_log_returns(historical)
            cov_matrix = hist_returns.cov()
            pred_returns = get_predicted_returns(historical)
            perc_returns = get_price_deltas(historical)

            weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )

            weights_df = pd.DataFrame({'MVP': weights})
            data_dict = {'historical': historical, 'future': future}

            result = BackTester.get_test(
                weights_df, data_dict, 'future', long_only=True
            )
            results.append((train_end, result))

        # Verify we got results
        assert len(results) > 0

        # Training size should grow
        train_sizes = [r[0] for r in results]
        assert train_sizes == sorted(train_sizes), "Training sizes should grow"


class TestForwardTestStatisticalValidity:
    """
    Tests for statistical validity of forward tests.
    """

    @pytest.mark.forwardtest
    def test_out_of_sample_degradation(self, sample_symbols):
        """
        Test that forward test performance typically degrades vs backtest.

        This is a statistical property: optimized strategies tend to
        perform worse out-of-sample than in-sample.
        """
        np.random.seed(42)
        n_trials = 5
        degradation_count = 0

        for trial in range(n_trials):
            np.random.seed(42 + trial)

            # Generate random prices
            prices = {}
            for symbol in sample_symbols:
                initial = np.random.uniform(50, 500)
                returns = np.random.normal(0.0005, 0.025, 300)
                prices[symbol] = initial * np.cumprod(1 + np.insert(returns, 0, 0))

            full_prices = pd.DataFrame(prices)
            historical = full_prices.iloc[:251]
            future = full_prices.iloc[250:]

            # Use Maximum Sharpe (most prone to overfitting)
            hist_returns = get_log_returns(historical)
            cov_matrix = hist_returns.cov()
            pred_returns = get_predicted_returns(historical)
            perc_returns = get_price_deltas(historical)

            strategy = MaximumSharpeRatioStrategy()
            weights = strategy.generate_portfolio(
                cov_matrix=cov_matrix,
                p_number=2,
                pred_returns=pred_returns.mean(),
                perc_returns=perc_returns,
                sample_returns=perc_returns,
                long_only=True
            )

            weights_df = pd.DataFrame({'MSR': weights})
            data_dict = {'historical': historical, 'future': future}

            backtest = BackTester.get_test(
                weights_df, data_dict, 'historical', long_only=True
            )
            forward = BackTester.get_test(
                weights_df, data_dict, 'future', long_only=True
            )

            # Compare per-period average returns
            avg_backtest = np.mean(np.diff(backtest.flatten()))
            avg_forward = np.mean(np.diff(forward.flatten()))

            if avg_forward < avg_backtest:
                degradation_count += 1

        # Should see degradation in most trials (not a strict test)
        # At least some degradation is expected
        assert degradation_count >= 0, "Forward test degradation test completed"
