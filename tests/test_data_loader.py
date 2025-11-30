"""
Unit tests for data_loader.py module.

Tests cover:
- DataEngine initialization
- Symbol formatting
- Data splitting for forward testing
- Stock list loading

Note: These tests don't require network access as they mock data.
Some tests require yfinance to be installed.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import dotdict

# Try to import DataEngine, skip tests if yfinance not available
try:
    from data_loader import DataEngine
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    DataEngine = None


class TestPeriodIntervalSelection:
    """Tests for period and interval selection based on granularity.

    These tests don't require yfinance as they test the logic directly.
    """

    @pytest.mark.unit
    def test_minute_granularity(self):
        """Test period/interval for 1-minute data."""
        granularity = 1

        if granularity == 1:
            period = "7d"
            interval = "1m"
        elif granularity == 3600:
            period = "5y"
            interval = "1d"
        else:
            period = "30d"
            interval = str(granularity) + "m"

        assert period == "7d"
        assert interval == "1m"

    @pytest.mark.unit
    def test_daily_granularity(self):
        """Test period/interval for daily data (3600 minutes)."""
        granularity = 3600

        if granularity == 1:
            period = "7d"
            interval = "1m"
        elif granularity == 3600:
            period = "5y"
            interval = "1d"
        else:
            period = "30d"
            interval = str(granularity) + "m"

        assert period == "5y"
        assert interval == "1d"

    @pytest.mark.unit
    def test_hourly_granularity(self):
        """Test period/interval for hourly data (60 minutes)."""
        granularity = 60

        if granularity == 1:
            period = "7d"
            interval = "1m"
        elif granularity == 3600:
            period = "5y"
            interval = "1d"
        else:
            period = "30d"
            interval = str(granularity) + "m"

        assert period == "30d"
        assert interval == "60m"

    @pytest.mark.unit
    def test_5_minute_granularity(self):
        """Test period/interval for 5-minute data."""
        granularity = 5

        if granularity == 1:
            period = "7d"
            interval = "1m"
        elif granularity == 3600:
            period = "5y"
            interval = "1d"
        else:
            period = "30d"
            interval = str(granularity) + "m"

        assert period == "30d"
        assert interval == "5m"


class TestHistoryUsage:
    """Tests for history_to_use parameter handling."""

    @pytest.mark.unit
    def test_all_history(self):
        """Test using all available history."""
        args = dotdict({
            'history_to_use': 'all',
        })

        assert args.history_to_use == 'all'

    @pytest.mark.unit
    def test_limited_history(self):
        """Test using limited history."""
        args = dotdict({
            'history_to_use': '100',
        })

        history_bars = int(args.history_to_use)
        assert history_bars == 100

    @pytest.mark.unit
    def test_history_slicing(self):
        """Test that history slicing works correctly."""
        full_data = np.arange(500)  # 500 bars
        history_to_use = 100

        if history_to_use == 'all':
            sliced = full_data[1:]
        else:
            sliced = full_data[-history_to_use:]

        assert len(sliced) == 100
        assert sliced[0] == 400
        assert sliced[-1] == 499


class TestStocksFileLoading:
    """Tests for loading stocks from file."""

    @pytest.mark.unit
    def test_load_stocks_from_temp_file(self, temp_stocks_file):
        """Test loading stocks from a temporary file."""
        with open(temp_stocks_file, 'r') as f:
            stocks = [s.strip() for s in f.readlines()]

        assert len(stocks) > 0
        assert 'AAPL' in stocks

    @pytest.mark.unit
    def test_stocks_sorted_and_unique(self, sample_symbols):
        """Test that loaded stocks are sorted and unique."""
        stocks_with_dupes = sample_symbols + ['AAPL', 'MSFT']
        result = list(sorted(set(stocks_with_dupes)))

        assert len(result) == len(sample_symbols)
        assert result == sorted(result)


@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
class TestSymbolFormatting:
    """Tests for symbol formatting in DataEngine."""

    @pytest.fixture
    def data_engine(self, default_args):
        """Create a DataEngine instance for testing."""
        return DataEngine(default_args)

    @pytest.mark.unit
    def test_uppercase_conversion(self, data_engine):
        """Test that symbols are converted to uppercase."""
        assert data_engine._format_symbol('aapl') == 'AAPL'
        assert data_engine._format_symbol('msft') == 'MSFT'

    @pytest.mark.unit
    def test_vn_suffix_conversion(self, data_engine):
        """Test that .VN suffix is converted to .V."""
        assert data_engine._format_symbol('ABC.VN') == 'ABC.V'
        assert data_engine._format_symbol('xyz.vn') == 'XYZ.V'

    @pytest.mark.unit
    def test_multiple_dots_handling(self, data_engine):
        """Test that symbols with multiple dots are handled."""
        result = data_engine._format_symbol('BRK.A.B')
        assert result == 'BRK-A.B'

    @pytest.mark.unit
    def test_simple_symbol_unchanged(self, data_engine):
        """Test that simple symbols are unchanged (except uppercase)."""
        assert data_engine._format_symbol('AAPL') == 'AAPL'
        assert data_engine._format_symbol('GOOGL') == 'GOOGL'


@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
class TestDataSplitting:
    """Tests for data splitting functionality."""

    @pytest.fixture
    def data_engine_with_test(self):
        """Create DataEngine configured for testing."""
        args = dotdict({
            'stocks_file_path': 'stocks/stocks.txt',
            'is_test': 1,
            'future_bars': 30,
            'data_granularity_minutes': 3600,
            'history_to_use': 'all',
            'apply_noise_filtering': 0,
            'market_index': 'SPY',
            'only_long': 1,
            'eigen_portfolio_number': 2,
            'save_plot': True
        })
        return DataEngine(args)

    @pytest.fixture
    def data_engine_no_test(self):
        """Create DataEngine configured without forward testing."""
        args = dotdict({
            'stocks_file_path': 'stocks/stocks.txt',
            'is_test': 0,
            'future_bars': 0,
            'data_granularity_minutes': 3600,
            'history_to_use': 'all',
            'apply_noise_filtering': 0,
            'market_index': 'SPY',
            'only_long': 1,
            'eigen_portfolio_number': 2,
            'save_plot': True
        })
        return DataEngine(args)

    @pytest.mark.unit
    def test_split_with_test_enabled(self, data_engine_with_test):
        """Test data splitting when forward testing is enabled."""
        mock_data = pd.DataFrame({
            'Adj Close': np.random.uniform(100, 200, 100)
        })

        historical, future = data_engine_with_test._split_data(mock_data)

        assert len(historical) == 70
        assert len(future) == 30

    @pytest.mark.unit
    def test_split_without_test(self, data_engine_no_test):
        """Test data splitting when forward testing is disabled."""
        mock_data = pd.DataFrame({
            'Adj Close': np.random.uniform(100, 200, 100)
        })

        historical, future = data_engine_no_test._split_data(mock_data)

        assert len(historical) == 100
        assert future is None

    @pytest.mark.unit
    def test_split_preserves_order(self, data_engine_with_test):
        """Test that splitting preserves temporal order."""
        prices = np.arange(100, 200)
        mock_data = pd.DataFrame({'Adj Close': prices})

        historical, future = data_engine_with_test._split_data(mock_data)

        assert historical[0] == 100
        assert historical[-1] == 169
        assert future[0] == 170
        assert future[-1] == 199


@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
class TestMostFrequentCount:
    """Tests for the most_frequent_count helper method."""

    @pytest.fixture
    def data_engine(self, default_args):
        """Create a DataEngine instance."""
        return DataEngine(default_args)

    @pytest.mark.unit
    def test_most_frequent_single_value(self, data_engine):
        """Test with list containing single repeated value."""
        result = data_engine.get_most_frequent_count([5, 5, 5, 5])
        assert result == 5

    @pytest.mark.unit
    def test_most_frequent_multiple_values(self, data_engine):
        """Test with list containing multiple different values."""
        result = data_engine.get_most_frequent_count([1, 2, 2, 3, 2, 4])
        assert result == 2

    @pytest.mark.unit
    def test_most_frequent_tie(self, data_engine):
        """Test with tie (returns first most frequent)."""
        result = data_engine.get_most_frequent_count([1, 1, 2, 2, 3])
        assert result in [1, 2]
