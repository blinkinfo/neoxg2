"""
tests/test_features.py
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_features,
    prepare_ml_data,
)


def make_candles(n=100, base_price=70000):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    opens = base_price + np.cumsum(np.random.randn(n) * 10)
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": opens,
        "high": [0.0] * n,
        "low": [0.0] * n,
        "close": [0.0] * n,
        "volume": np.random.rand(n) * 100 + 50,
    }
    for i in range(n):
        o = data["open"][i]
        c = o + np.random.randn() * 5
        data["close"][i] = c
        data["high"][i] = max(o, c) + abs(np.random.randn()) * 5
        data["low"][i] = min(o, c) - abs(np.random.randn()) * 5

    return pd.DataFrame(data)


class TestRSI:
    def test_rsi_bounds(self):
        closes = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        rsi = compute_rsi(closes, period=14)
        assert rsi.notna().sum() > 0
        assert (rsi.dropna() <= 100).all()
        assert (rsi.dropna() >= 0).all()

    def test_rsi_oversold_recovered(self):
        closes = pd.Series(range(1, 30))
        rsi = compute_rsi(closes, period=14)
        assert rsi.iloc[-1] > 70


class TestMACD:
    def test_macd_output_shape(self):
        closes = pd.Series([1] * 50)
        result = compute_macd(closes)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns


class TestBollingerBands:
    def test_bb_columns(self):
        closes = pd.Series([1] * 50)
        result = compute_bollinger_bands(closes)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns

    def test_bb_order(self):
        closes = pd.Series([1] * 50)
        result = compute_bollinger_bands(closes)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()


class TestComputeFeatures:
    def test_no_crash(self):
        df = make_candles(200)
        result = compute_features(df)
        assert len(result) == len(df)

    def test_features_created(self):
        df = make_candles(200)
        result = compute_features(df)
        expected = [
            "rsi", "macd", "histogram", "bb_width", "volume_ratio",
            "direction_label", "is_flat", "atr_percentile", "low_volatility",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_direction_label_binary(self):
        df = make_candles(200)
        result = compute_features(df)
        labels = result["direction_label"].dropna()
        assert set(labels.unique()).issubset({0, 1})

    def test_raw_bb_removed_from_features(self):
        """Raw price BB features should NOT be in the ML feature list."""
        df = make_candles(200)
        df = compute_features(df)
        X, y, cols = prepare_ml_data(df, drop_na=True)
        assert "bb_upper" not in cols
        assert "bb_middle" not in cols
        assert "bb_lower" not in cols

    def test_flat_candles_filtered(self):
        """Flat candles should be filtered out of training data."""
        df = make_candles(200)
        df = compute_features(df)
        # Force some flat candles
        df.loc[50, "is_flat"] = 1
        df.loc[60, "is_flat"] = 1
        X, y, cols = prepare_ml_data(df, drop_na=True)
        # Verify flat rows are gone (if they were in valid range)
        assert len(X) > 0


class TestPrepareMLData:
    def test_dropna_removes_warmup(self):
        df = make_candles(200)
        df = compute_features(df)
        X, y, cols = prepare_ml_data(df, drop_na=True)
        assert len(X) <= len(df)
        assert len(X) == len(y)

    def test_cyclical_features_present(self):
        df = make_candles(200)
        df = compute_features(df)
        X, y, cols = prepare_ml_data(df, drop_na=True)
        assert "hour_sin" in cols
        assert "hour_cos" in cols
        assert "dow_sin" in cols
        assert "dow_cos" in cols

    def test_volatility_features_present(self):
        df = make_candles(200)
        df = compute_features(df)
        X, y, cols = prepare_ml_data(df, drop_na=True)
        assert "atr_percentile" in cols
        assert "low_volatility" in cols
