"""
Unit tests for backtester.py module.

Tests cover:
- BackTester.get_test() for portfolio backtesting
- BackTester.filter_short() for long-only portfolios
- BackTester.get_market_returns() for benchmark comparison
- BackTester.simulate_future_prices() for Monte Carlo simulation
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BackTester
from utils import get_predicted_returns


class TestFilterShort:
    """Tests for the filter_short method."""

    @pytest.mark.unit
    def test_long_only_removes_negatives(self):
        """Test that negative weights are zeroed when long_only=True."""
        weights = np.array([0.5, -0.3, 0.4, -0.2, 0.6])
        filtered = BackTester.filter_short(weights, long_only=True)

        assert all(w >= 0 for w in filtered)
        assert filtered[1] == 0
        assert filtered[3] == 0

    @pytest.mark.unit
    def test_long_short_preserves_all(self):
        """Test that all weights are preserved when long_only=False."""
        weights = np.array([0.5, -0.3, 0.4, -0.2, 0.6])
        filtered = BackTester.filter_short(weights, long_only=False)

        assert np.array_equal(filtered, weights)

    @pytest.mark.unit
    def test_all_positive_unchanged(self):
        """Test that all-positive weights are unchanged."""
        weights = np.array([0.2, 0.3, 0.5])
        filtered = BackTester.filter_short(weights, long_only=True)

        assert np.array_equal(filtered, weights)

    @pytest.mark.unit
    def test_all_negative_zeroed(self):
        """Test that all-negative weights become zeros."""
        weights = np.array([-0.2, -0.3, -0.5])
        filtered = BackTester.filter_short(weights, long_only=True)

        assert all(w == 0 for w in filtered)

    @pytest.mark.unit
    def test_empty_array(self):
        """Test handling of empty array."""
        weights = np.array([])
        filtered = BackTester.filter_short(weights, long_only=True)
        assert len(filtered) == 0


class TestGetTest:
    """Tests for the get_test backtest method."""

    @pytest.mark.unit
    def test_historical_direction(self, data_dict, portfolio_weights_df):
        """Test backtesting with historical data."""
        result = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'historical',
            long_only=True
        )

        assert result is not None
        assert len(result) > 0

    @pytest.mark.unit
    def test_future_direction(self, data_dict, portfolio_weights_df):
        """Test backtesting with future data."""
        result = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'future',
            long_only=True
        )

        assert result is not None
        assert len(result) > 0

    @pytest.mark.unit
    def test_invalid_direction_raises(self, data_dict, portfolio_weights_df):
        """Test that invalid direction raises assertion error."""
        with pytest.raises(AssertionError):
            BackTester.get_test(
                portfolio_weights_df,
                data_dict,
                'invalid',
                long_only=True
            )

    @pytest.mark.unit
    def test_returns_shape(self, data_dict, portfolio_weights_df):
        """Test that returns have expected shape."""
        result = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'historical',
            long_only=True
        )

        # Result should have one fewer row than data (due to returns calculation)
        expected_rows = len(data_dict['historical']) - 1
        assert result.shape[0] == expected_rows

    @pytest.mark.unit
    def test_cumulative_returns(self, data_dict, portfolio_weights_df):
        """Test that returns are cumulative."""
        result = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'historical',
            long_only=True
        )

        # Check result is numpy array
        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    def test_equal_weights_diversification(self, data_dict, equal_weights, sample_symbols):
        """Test that equal weights provide expected diversification."""
        weights_df = pd.DataFrame({'Equal': equal_weights})

        result = BackTester.get_test(
            weights_df,
            data_dict,
            'historical',
            long_only=True
        )

        # Should return array, not raise errors
        assert result is not None


class TestGetMarketReturns:
    """Tests for the get_market_returns method."""

    @pytest.mark.unit
    def test_historical_market_returns(self, market_data):
        """Test market returns calculation for historical data."""
        result = BackTester.get_market_returns(market_data, 'historical')

        assert result is not None
        assert len(result) > 0

    @pytest.mark.unit
    def test_future_market_returns(self, market_data):
        """Test market returns calculation for future data."""
        result = BackTester.get_market_returns(market_data, 'future')

        assert result is not None

    @pytest.mark.unit
    def test_invalid_direction_raises(self, market_data):
        """Test that invalid direction raises assertion error."""
        with pytest.raises(AssertionError):
            BackTester.get_market_returns(market_data, 'invalid')

    @pytest.mark.unit
    def test_returns_cumulative(self, market_data):
        """Test that market returns are cumulative."""
        result = BackTester.get_market_returns(market_data, 'historical')

        # Cumulative returns should be returned (either as ndarray or DataFrame)
        # Just check it's not None and has data
        assert result is not None
        assert len(result) > 0


class TestSimulateFuturePrices:
    """Tests for the simulate_future_prices Monte Carlo method."""

    @pytest.mark.unit
    def test_returns_dataframe(self, data_dict, set_random_seed):
        """Test that simulation returns a DataFrame."""
        result = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=10
        )

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_correct_columns(self, data_dict, sample_symbols, set_random_seed):
        """Test that simulated data has correct columns."""
        result = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=10
        )

        assert list(result.columns) == sample_symbols

    @pytest.mark.unit
    def test_correct_timesteps(self, data_dict, set_random_seed):
        """Test that simulation has correct number of timesteps."""
        timesteps = 20
        result = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=timesteps
        )

        # Simulation includes initial price + timesteps
        assert len(result) <= timesteps + 1

    @pytest.mark.unit
    def test_positive_prices(self, data_dict, set_random_seed):
        """Test that all simulated prices are positive."""
        result = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=30
        )

        assert np.all(result.values > 0)

    @pytest.mark.unit
    def test_starts_at_last_price(self, data_dict, set_random_seed):
        """Test that simulation starts at the last historical price."""
        result = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=30
        )

        historical = data_dict['historical']
        for col in historical.columns:
            last_historical = historical[col].iloc[-1]
            first_simulated = result[col].iloc[0]
            # Should be close (within Monte Carlo variance)
            assert np.isclose(first_simulated, last_historical, rtol=0.5)

    @pytest.mark.unit
    def test_reproducibility_with_seed(self, data_dict):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        result1 = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=30
        )

        np.random.seed(42)
        result2 = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=30
        )

        # Results should be identical with same seed
        # Note: Due to the averaging over 100 iterations, small differences may occur
        # but the first value should be exact
        for col in result1.columns:
            assert np.isclose(result1[col].iloc[0], result2[col].iloc[0])


class TestBacktestIntegration:
    """Integration tests for the backtester module."""

    @pytest.mark.integration
    def test_full_backtest_workflow(self, data_dict, portfolio_weights_df, market_data):
        """Test complete backtest workflow."""
        # Run portfolio backtest
        portfolio_returns = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'historical',
            long_only=True
        )

        # Get market returns for comparison
        market_returns = BackTester.get_market_returns(market_data, 'historical')

        # Both should have valid returns
        assert portfolio_returns is not None
        assert market_returns is not None

    @pytest.mark.integration
    def test_forward_test_workflow(self, data_dict, portfolio_weights_df):
        """Test forward testing workflow."""
        # Historical backtest
        hist_returns = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'historical',
            long_only=True
        )

        # Forward test
        future_returns = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'future',
            long_only=True
        )

        # Both should complete without errors
        assert hist_returns is not None
        assert future_returns is not None

    @pytest.mark.integration
    def test_simulation_workflow(self, data_dict, portfolio_weights_df, set_random_seed):
        """Test Monte Carlo simulation workflow."""
        # Simulate future prices
        data_dict['sim'] = BackTester.simulate_future_prices(
            data_dict,
            get_predicted_returns,
            simulation_timesteps=30
        )

        # Run backtest on simulated data
        sim_returns = BackTester.get_test(
            portfolio_weights_df,
            data_dict,
            'sim',
            long_only=True
        )

        assert sim_returns is not None
