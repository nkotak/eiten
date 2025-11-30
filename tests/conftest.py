"""
Pytest configuration and shared fixtures for Eiten tests.

This module provides reusable test fixtures that generate synthetic data
for testing portfolio strategies without relying on external data sources.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import dotdict


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def random_seed():
    """Provide a consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def set_random_seed(random_seed):
    """Set numpy random seed for reproducibility."""
    np.random.seed(random_seed)
    return random_seed


# =============================================================================
# SYNTHETIC DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_symbols():
    """List of sample stock symbols for testing."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']


@pytest.fixture
def num_trading_days():
    """Number of trading days for synthetic data generation."""
    return 252  # One trading year


@pytest.fixture
def future_bars():
    """Number of bars reserved for forward testing."""
    return 30


@pytest.fixture
def synthetic_prices(sample_symbols, num_trading_days, set_random_seed):
    """
    Generate synthetic price data using geometric Brownian motion.

    This creates realistic-looking price series without hardcoding values.
    Each symbol gets different random parameters for volatility and drift.
    """
    prices = {}

    for symbol in sample_symbols:
        # Random but reproducible parameters per symbol
        initial_price = np.random.uniform(50, 500)
        daily_drift = np.random.uniform(-0.0005, 0.001)  # -0.05% to 0.1% daily
        daily_volatility = np.random.uniform(0.01, 0.03)  # 1% to 3% daily

        # Generate log returns
        log_returns = np.random.normal(
            daily_drift,
            daily_volatility,
            num_trading_days
        )

        # Convert to prices using geometric Brownian motion
        cumulative_returns = np.cumsum(log_returns)
        prices[symbol] = initial_price * np.exp(cumulative_returns)

    return pd.DataFrame(prices)


@pytest.fixture
def historical_prices(synthetic_prices, future_bars):
    """Historical price data (excluding forward test period)."""
    return synthetic_prices.iloc[:-future_bars].copy()


@pytest.fixture
def future_prices(synthetic_prices, future_bars):
    """Future price data (for forward testing)."""
    return synthetic_prices.iloc[-future_bars:].copy()


@pytest.fixture
def data_dict(historical_prices, future_prices):
    """
    Data dictionary in the format expected by Eiten modules.
    """
    return {
        'historical': historical_prices,
        'future': future_prices
    }


# =============================================================================
# COVARIANCE AND RETURNS FIXTURES
# =============================================================================

@pytest.fixture
def log_returns(historical_prices):
    """Calculate log returns from historical prices."""
    return np.log(historical_prices / historical_prices.shift(1))[1:]


@pytest.fixture
def price_deltas(historical_prices):
    """Calculate percentage price changes."""
    return ((historical_prices - historical_prices.shift(1)) /
            historical_prices.shift(1))[1:]


@pytest.fixture
def covariance_matrix(log_returns):
    """Covariance matrix of log returns."""
    return log_returns.cov()


@pytest.fixture
def correlation_matrix(log_returns):
    """Correlation matrix of log returns."""
    return log_returns.corr()


# =============================================================================
# PORTFOLIO WEIGHT FIXTURES
# =============================================================================

@pytest.fixture
def equal_weights(sample_symbols):
    """Equal-weighted portfolio."""
    n = len(sample_symbols)
    return {symbol: 1.0 / n for symbol in sample_symbols}


@pytest.fixture
def random_weights(sample_symbols, set_random_seed):
    """Random portfolio weights that sum to 1."""
    n = len(sample_symbols)
    weights = np.random.random(n)
    weights = weights / weights.sum()
    return {symbol: w for symbol, w in zip(sample_symbols, weights)}


@pytest.fixture
def long_short_weights(sample_symbols, set_random_seed):
    """Random long-short portfolio weights."""
    n = len(sample_symbols)
    # Generate weights between -1 and 1
    weights = np.random.uniform(-1, 1, n)
    # Normalize positive and negative separately
    pos_mask = weights > 0
    neg_mask = weights < 0
    if pos_mask.any():
        weights[pos_mask] = weights[pos_mask] / weights[pos_mask].sum()
    if neg_mask.any():
        weights[neg_mask] = weights[neg_mask] / np.abs(weights[neg_mask]).sum()
    return {symbol: w for symbol, w in zip(sample_symbols, weights)}


@pytest.fixture
def portfolio_weights_df(equal_weights, random_weights):
    """DataFrame of portfolio weights for multiple strategies."""
    return pd.DataFrame({
        'Equal Weight': equal_weights,
        'Random': random_weights
    })


# =============================================================================
# ARGUMENT FIXTURES
# =============================================================================

@pytest.fixture
def default_args():
    """Default arguments for Eiten."""
    return dotdict({
        'stocks_file_path': 'stocks/stocks.txt',
        'is_test': 1,
        'future_bars': 30,
        'data_granularity_minutes': 3600,
        'history_to_use': 'all',
        'apply_noise_filtering': 1,
        'market_index': 'SPY',
        'only_long': 1,
        'eigen_portfolio_number': 2,
        'save_plot': True
    })


@pytest.fixture
def test_args_daily():
    """Arguments for daily data testing."""
    return dotdict({
        'stocks_file_path': 'stocks/stocks.txt',
        'is_test': 1,
        'future_bars': 30,
        'data_granularity_minutes': 3600,  # Daily
        'history_to_use': 'all',
        'apply_noise_filtering': 0,
        'market_index': 'SPY',
        'only_long': 1,
        'eigen_portfolio_number': 2,
        'save_plot': True
    })


@pytest.fixture
def test_args_intraday():
    """Arguments for intraday data testing."""
    return dotdict({
        'stocks_file_path': 'stocks/stocks.txt',
        'is_test': 0,
        'future_bars': 0,
        'data_granularity_minutes': 60,  # Hourly
        'history_to_use': 100,
        'apply_noise_filtering': 0,
        'market_index': 'QQQ',
        'only_long': 1,
        'eigen_portfolio_number': 3,
        'save_plot': True
    })


# =============================================================================
# MARKET DATA FIXTURES
# =============================================================================

@pytest.fixture
def market_prices(num_trading_days, set_random_seed):
    """Generate synthetic market index prices."""
    initial_price = 400.0
    daily_drift = 0.0003  # ~8% annual return
    daily_volatility = 0.012  # ~19% annual volatility

    log_returns = np.random.normal(daily_drift, daily_volatility, num_trading_days)
    cumulative_returns = np.cumsum(log_returns)
    prices = initial_price * np.exp(cumulative_returns)

    return pd.DataFrame({'SPY': prices})


@pytest.fixture
def market_data(market_prices, future_bars):
    """Market data dictionary with historical and future splits."""
    return {
        'historical': market_prices.iloc[:-future_bars],
        'future': market_prices.iloc[-future_bars:]
    }


# =============================================================================
# STRATEGY PARAMETER FIXTURES
# =============================================================================

@pytest.fixture
def strategy_params(covariance_matrix, price_deltas, log_returns):
    """Common parameters passed to portfolio strategies."""
    return {
        'cov_matrix': covariance_matrix,
        'p_number': 2,
        'pred_returns': price_deltas.mean(),
        'perc_returns': price_deltas,
        'sample_returns': price_deltas,
        'long_only': True
    }


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def known_returns():
    """
    Create returns with known statistical properties for validation.

    This fixture creates data where we know the expected outcomes,
    making it easier to validate calculations.
    """
    # Create simple returns where we can calculate expected portfolio returns
    np.random.seed(42)
    n_assets = 3
    n_periods = 100

    # Create returns with known mean and covariance
    means = np.array([0.001, 0.002, 0.0015])  # Daily returns
    cov = np.array([
        [0.0004, 0.0001, 0.00015],
        [0.0001, 0.0009, 0.0002],
        [0.00015, 0.0002, 0.0006]
    ])

    returns = np.random.multivariate_normal(means, cov, n_periods)

    return pd.DataFrame(
        returns,
        columns=['Stock_A', 'Stock_B', 'Stock_C']
    )


@pytest.fixture
def temp_stocks_file(tmp_path, sample_symbols):
    """Create a temporary stocks file for testing."""
    stocks_file = tmp_path / "test_stocks.txt"
    stocks_file.write_text('\n'.join(sample_symbols))
    return str(stocks_file)


# =============================================================================
# VALIDATION FIXTURES
# =============================================================================

@pytest.fixture
def tolerance():
    """Numerical tolerance for floating point comparisons."""
    return 1e-10


@pytest.fixture
def high_tolerance():
    """Higher tolerance for accumulated numerical errors."""
    return 1e-6
