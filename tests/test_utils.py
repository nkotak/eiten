"""
Unit tests for utils.py module.

Tests cover:
- normalize_weights function (including edge cases)
- get_price_deltas function
- get_log_returns function
- get_predicted_returns function
- random_matrix_theory_based_cov function
- dotdict class
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    normalize_weights,
    get_price_deltas,
    get_log_returns,
    get_predicted_returns,
    get_exp_returns,
    random_matrix_theory_based_cov,
    dotdict
)


class TestDotdict:
    """Tests for the dotdict helper class."""

    @pytest.mark.unit
    def test_dotdict_access(self):
        """Test dot notation access to dictionary values."""
        d = dotdict({'key1': 'value1', 'key2': 42})
        assert d.key1 == 'value1'
        assert d.key2 == 42

    @pytest.mark.unit
    def test_dotdict_set(self):
        """Test setting values via dot notation."""
        d = dotdict({})
        d.new_key = 'new_value'
        assert d['new_key'] == 'new_value'
        assert d.new_key == 'new_value'

    @pytest.mark.unit
    def test_dotdict_delete(self):
        """Test deleting values via dot notation."""
        d = dotdict({'key': 'value'})
        del d.key
        assert 'key' not in d

    @pytest.mark.unit
    def test_dotdict_missing_key(self):
        """Test accessing missing key returns None (dict.get behavior)."""
        d = dotdict({})
        assert d.nonexistent is None


class TestNormalizeWeights:
    """Tests for the normalize_weights function."""

    @pytest.mark.unit
    def test_all_positive_weights(self):
        """Test normalization when all weights are positive."""
        weights = [0.3, 0.5, 0.2]
        normalized = normalize_weights(weights.copy())
        assert np.isclose(sum(normalized), 1.0), "Positive weights should sum to 1"

    @pytest.mark.unit
    def test_all_negative_weights(self):
        """Test normalization when all weights are negative."""
        weights = [-0.3, -0.5, -0.2]
        normalized = normalize_weights(weights.copy())
        assert np.isclose(sum(normalized), -1.0), "Negative weights should sum to -1"

    @pytest.mark.unit
    def test_mixed_weights(self):
        """Test normalization with mixed positive and negative weights."""
        weights = [0.6, -0.3, 0.4, -0.2]
        normalized = normalize_weights(weights.copy())

        pos_sum = sum(w for w in normalized if w > 0)
        neg_sum = sum(w for w in normalized if w < 0)

        assert np.isclose(pos_sum, 1.0), "Positive weights should sum to 1"
        assert np.isclose(neg_sum, -1.0), "Negative weights should sum to -1"

    @pytest.mark.unit
    def test_single_weight(self):
        """Test normalization of a single weight."""
        weights = [5.0]
        normalized = normalize_weights(weights.copy())
        assert np.isclose(normalized[0], 1.0)

    @pytest.mark.unit
    def test_zero_weights_preserved(self):
        """Test that zero weights remain zero after normalization."""
        weights = [0.5, 0.0, 0.5]
        normalized = normalize_weights(weights.copy())
        assert normalized[1] == 0.0

    @pytest.mark.unit
    def test_empty_weights(self):
        """Test normalization of empty array."""
        weights = []
        normalized = normalize_weights(weights)
        assert len(normalized) == 0

    @pytest.mark.unit
    def test_numpy_array_input(self):
        """Test that function works with numpy arrays."""
        weights = np.array([0.2, 0.3, 0.5])
        normalized = normalize_weights(weights.copy())
        assert isinstance(normalized, np.ndarray)
        assert np.isclose(sum(normalized), 1.0)

    @pytest.mark.unit
    def test_large_weights(self):
        """Test normalization of large weight values."""
        weights = [1000.0, 2000.0, 3000.0]
        normalized = normalize_weights(weights.copy())
        assert np.isclose(sum(normalized), 1.0)

    @pytest.mark.unit
    def test_small_weights(self):
        """Test normalization of very small weight values."""
        weights = [0.00001, 0.00002, 0.00003]
        normalized = normalize_weights(weights.copy())
        assert np.isclose(sum(normalized), 1.0)


class TestGetPriceDeltas:
    """Tests for the get_price_deltas function."""

    @pytest.mark.unit
    def test_basic_price_delta(self):
        """Test basic percentage change calculation."""
        prices = pd.DataFrame({'A': [100.0, 110.0, 99.0]})
        deltas = get_price_deltas(prices)

        expected = [0.1, -0.1]  # 10% up, ~9.09% down
        assert len(deltas) == 2
        assert np.isclose(deltas['A'].iloc[0], 0.1)
        assert np.isclose(deltas['A'].iloc[1], -0.1, rtol=0.01)

    @pytest.mark.unit
    def test_multiple_columns(self, historical_prices):
        """Test price deltas with multiple columns."""
        deltas = get_price_deltas(historical_prices)

        assert deltas.shape[0] == historical_prices.shape[0] - 1
        assert deltas.shape[1] == historical_prices.shape[1]
        assert list(deltas.columns) == list(historical_prices.columns)

    @pytest.mark.unit
    def test_returns_dataframe(self, historical_prices):
        """Test that output is a DataFrame."""
        deltas = get_price_deltas(historical_prices)
        assert isinstance(deltas, pd.DataFrame)

    @pytest.mark.unit
    def test_no_inf_values(self, historical_prices):
        """Test that no infinite values are produced."""
        deltas = get_price_deltas(historical_prices)
        assert not np.any(np.isinf(deltas.values))


class TestGetLogReturns:
    """Tests for the get_log_returns function."""

    @pytest.mark.unit
    def test_basic_log_return(self):
        """Test basic log return calculation."""
        prices = pd.DataFrame({'A': [100.0, 110.0]})
        log_ret = get_log_returns(prices)

        expected = np.log(110.0 / 100.0)
        assert np.isclose(log_ret['A'].iloc[0], expected)

    @pytest.mark.unit
    def test_log_returns_shape(self, historical_prices):
        """Test that log returns have correct shape."""
        log_ret = get_log_returns(historical_prices)
        assert log_ret.shape[0] == historical_prices.shape[0] - 1

    @pytest.mark.unit
    def test_log_returns_symmetric(self):
        """Test that log returns are approximately symmetric for small changes."""
        prices = pd.DataFrame({'A': [100.0, 101.0, 100.0]})
        log_ret = get_log_returns(prices)

        # For small changes, log returns should be approximately symmetric
        assert np.isclose(log_ret['A'].iloc[0], -log_ret['A'].iloc[1], rtol=0.02)

    @pytest.mark.unit
    def test_no_nan_values(self, historical_prices):
        """Test that no NaN values are in the result (except first row)."""
        log_ret = get_log_returns(historical_prices)
        assert not np.any(np.isnan(log_ret.values))


class TestGetPredictedReturns:
    """Tests for the get_predicted_returns function."""

    @pytest.mark.unit
    def test_predicted_returns_shape(self, historical_prices):
        """Test that predicted returns have correct shape."""
        pred = get_predicted_returns(historical_prices)
        assert pred.shape[0] == historical_prices.shape[0] - 1

    @pytest.mark.unit
    def test_predicted_returns_weighting(self):
        """Test that more recent returns get higher weights."""
        prices = pd.DataFrame({'A': [100.0, 102.0, 101.0, 103.0, 105.0]})
        pred = get_predicted_returns(prices)

        # Recent returns should have larger absolute values due to lower divisors
        assert abs(pred['A'].iloc[-1]) > abs(pred['A'].iloc[0])


class TestGetExpReturns:
    """Tests for the get_exp_returns function."""

    @pytest.mark.unit
    def test_exp_returns_shape(self, historical_prices):
        """Test that exponential returns have correct shape."""
        exp_ret = get_exp_returns(historical_prices)
        assert exp_ret.shape[0] == historical_prices.shape[0] - 1


class TestRandomMatrixTheoryBasedCov:
    """Tests for the random_matrix_theory_based_cov function."""

    @pytest.mark.unit
    def test_output_is_dataframe(self, log_returns):
        """Test that output is a DataFrame."""
        filtered_cov = random_matrix_theory_based_cov(log_returns)
        assert isinstance(filtered_cov, pd.DataFrame)

    @pytest.mark.unit
    def test_preserves_columns(self, log_returns):
        """Test that column names are preserved."""
        filtered_cov = random_matrix_theory_based_cov(log_returns)
        assert list(filtered_cov.columns) == list(log_returns.columns)

    @pytest.mark.unit
    def test_symmetric_matrix(self, log_returns):
        """Test that filtered covariance matrix is symmetric."""
        filtered_cov = random_matrix_theory_based_cov(log_returns)
        diff = np.abs(filtered_cov.values - filtered_cov.values.T)
        assert np.all(diff < 1e-10)

    @pytest.mark.unit
    def test_positive_diagonal(self, log_returns):
        """Test that diagonal elements are positive (variances)."""
        filtered_cov = random_matrix_theory_based_cov(log_returns)
        diagonal = np.diag(filtered_cov.values)
        assert np.all(diagonal >= 0)

    @pytest.mark.unit
    def test_filters_noise(self, log_returns):
        """Test that filtering reduces some eigenvalues."""
        original_cov = log_returns.cov()
        filtered_cov = random_matrix_theory_based_cov(log_returns)

        orig_eigenvalues = np.linalg.eigvalsh(original_cov)
        filt_eigenvalues = np.linalg.eigvalsh(filtered_cov)

        # Some eigenvalues should be reduced (filtered)
        # This is a weak test since exact behavior depends on data
        assert not np.allclose(orig_eigenvalues, filt_eigenvalues)


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    @pytest.mark.unit
    def test_returns_chain(self, synthetic_prices):
        """Test that returns functions can be chained."""
        log_ret = get_log_returns(synthetic_prices)
        price_deltas = get_price_deltas(synthetic_prices)

        # Both should have same shape
        assert log_ret.shape == price_deltas.shape

        # For small returns, log returns ~ price returns
        assert np.allclose(
            log_ret.values,
            price_deltas.values,
            rtol=0.1,  # 10% relative tolerance for approximation
            atol=0.01
        )

    @pytest.mark.unit
    def test_covariance_from_different_returns(self, synthetic_prices):
        """Test covariance calculation from different return types."""
        log_ret = get_log_returns(synthetic_prices)
        price_deltas = get_price_deltas(synthetic_prices)

        cov_log = log_ret.cov()
        cov_delta = price_deltas.cov()

        # Covariances should be similar for small returns
        assert cov_log.shape == cov_delta.shape
