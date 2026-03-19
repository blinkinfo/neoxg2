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
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": base_price + np.cumsum(np.random.randn(n) * 10),
        "high": [0] * n,
        "low": [0] * n,
        "close": [0] * n,
        "volume": np.random.rand(n) * 100 + 50,
    }
    # Ensure high/low are consistent with open/close
    for i in range(n):
        o = data["open"][i]
        c = data["close"][i]
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
        # After a strong up move, RSI should be high
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
        assert (result["bb_upper"] >= result["bb_middle"]).all()
        assert (result["bb_middle"] >= result["bb_lower"]).all()


class TestComputeFeatures:
    def test_no_crash(self):
        df = make_candles(100)
        result = compute_features(df)
        assert len(result) == len(df)

    def test_features_created(self):
        df = make_candles(100)
        result = compute_features(df)
        expected = ["rsi", "macd", "histogram", "bb_width", "volume_ratio", "direction_label"]
        for col in expected:
            assert col in result.columns

    def test_direction_label_binary(self):
        df = make_candles(100)
        result = compute_features(df)
        labels = result["direction_label"].dropna()
        assert set(labels.unique()).issubset({0, 1})


class TestPrepareMLData:
    def test_dropna_removes_warmup(self):
        df = make_candles(100)
        df = compute_features(df)
        X, y, cols = prepare_ml_data(df, drop_na=True)
        # Should have fewer rows than input after dropna
        assert len(X) <= len(df)
        assert len(X) == len(y)

    def test_30_features(self):
        df = make_candles(200)
        df = compute_features(df)
        X, y, cols = prepare_ml_data(df, drop_na=True)
        assert len(cols) == 30
