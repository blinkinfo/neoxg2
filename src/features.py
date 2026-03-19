"""
features.py
Feature engineering for BTC 5-min candle direction prediction.
Computes technical indicators used as features for XGBoost model.
"""

import pandas as pd
import numpy as np
from src.config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, MOMENTUM_PERIODS, VOL_LOOKBACK
)


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Compute MACD line, signal line, and histogram.
    Returns DataFrame with macd, signal, histogram columns.
    """
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }, index=closes.index)


def compute_bollinger_bands(closes: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Compute Bollinger Bands (upper, middle, lower, width).
    """
    sma = closes.rolling(window=period, min_periods=period).mean()
    rolling_std = closes.rolling(window=period, min_periods=period).std()
    
    upper = sma + (rolling_std * std_dev)
    lower = sma - (rolling_std * std_dev)
    width = (upper - lower) / sma  # normalized width
    
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": sma,
        "bb_lower": lower,
        "bb_width": width
    }, index=closes.index)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for the given candle DataFrame.
    
    Input DataFrame must have columns: open, high, low, close, volume
    Output DataFrame adds all feature columns.
    """
    df = df.copy()
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_price = df["open"]
    volume = df["volume"]
    
    # ─── Price-based features ────────────────────────────────────────────
    
    # Candle body and wick sizes
    body = close - open_price
    upper_wick = high - pd.concat([close, open_price], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_price], axis=1).min(axis=1) - low
    candle_size = high - low
    
    df["candle_body"] = body
    df["candle_body_abs"] = body.abs()
    df["candle_body_pct"] = body.abs() / candle_size  # body as fraction of range
    df["upper_wick"] = upper_wick
    df["lower_wick"] = lower_wick
    df["candle_size"] = candle_size
    df["candle_size_pct"] = candle_size / close  # ATR-like
    
    # ─── RSI ────────────────────────────────────────────────────────────────
    rsi = compute_rsi(close, RSI_PERIOD)
    df["rsi"] = rsi
    
    # RSI overbought/oversold flags
    df["rsi_overbought"] = (rsi > 70).astype(int)
    df["rsi_oversold"] = (rsi < 30).astype(int)
    
    # ─── MACD ───────────────────────────────────────────────────────────────
    macd_df = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    for col in ["macd", "signal", "histogram"]:
        df[col] = macd_df[col]
    
    # MACD crossover signal
    df["macd_histogram_pos"] = (macd_df["histogram"] > 0).astype(int)
    
    # ─── Bollinger Bands ─────────────────────────────────────────────────────
    bb_df = compute_bollinger_bands(close, BB_PERIOD, BB_STD)
    for col in ["bb_upper", "bb_middle", "bb_lower", "bb_width"]:
        df[col] = bb_df[col]
    
    # Price relative to BB
    df["bb_position"] = (close - bb_df["bb_lower"]) / (bb_df["bb_upper"] - bb_df["bb_lower"])
    df["price_vs_bb_upper"] = (close - bb_df["bb_upper"]) / bb_df["bb_upper"]
    df["price_vs_bb_lower"] = (close - bb_df["bb_lower"]) / bb_df["bb_lower"]
    
    # ─── Momentum ────────────────────────────────────────────────────────────
    for period in MOMENTUM_PERIODS:
        df[f"momentum_{period}"] = close - close.shift(period)
        df[f"momentum_{period}_pct"] = (close - close.shift(period)) / close.shift(period)
    
    # ─── Volume ───────────────────────────────────────────────────────────────
    vol_ma = volume.rolling(window=VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()
    df["volume_ratio"] = volume / vol_ma  # spike = > 1.5, crush = < 0.5
    df["volume_ma"] = vol_ma
    
    # ─── Volatility (ATR) ─────────────────────────────────────────────────────
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14).mean()
    df["atr_pct"] = df["atr"] / close
    
    # ─── Direction labels ──────────────────────────────────────────────────────
    # Label: 1 = next candle closed UP, 0 = next candle closed DOWN
    # (this is the TARGET we train the model to predict)
    df["next_close"] = close.shift(-1)
    df["direction_label"] = (df["next_close"] > close).astype(int)
    
    # Also compute: was the CURRENT candle up or down?
    df["candle_up"] = (close > open_price).astype(int)
    
    # ─── Time features ───────────────────────────────────────────────────────
    # (for seasonality — crypto has some hourly/daily patterns)
    if "datetime" in df.columns:
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["dayofweek"] = df["datetime"].dt.dayofweek
    
    return df


def prepare_ml_data(df: pd.DataFrame, drop_na=True):
    """
    Prepare final feature matrix X and target vector y.
    
    Features used for training:
    - rsi, rsi_overbought, rsi_oversold
    - macd, signal, histogram, macd_histogram_pos
    - bb_upper, bb_middle, bb_lower, bb_width, bb_position
    - price_vs_bb_upper, price_vs_bb_lower
    - momentum_3, momentum_5, momentum_7 (and pct versions)
    - volume_ratio
    - atr, atr_pct
    - candle_body_pct, upper_wick, lower_wick
    - hour, minute, dayofweek
    - candle_up (previous candle direction)
    """
    feature_cols = [
        "rsi", "rsi_overbought", "rsi_oversold",
        "macd", "signal", "histogram", "macd_histogram_pos",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
        "price_vs_bb_upper", "price_vs_bb_lower",
        "momentum_3", "momentum_3_pct",
        "momentum_5", "momentum_5_pct",
        "momentum_7", "momentum_7_pct",
        "volume_ratio",
        "atr", "atr_pct",
        "candle_body_pct", "upper_wick", "lower_wick",
        "hour", "minute", "dayofweek",
        "candle_up",
    ]
    
    X = df[feature_cols].copy()
    y = df["direction_label"].copy()
    
    if drop_na:
        # Drop rows with NaN (first ~20 candles due to rolling windows)
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    
    return X, y, feature_cols


if __name__ == "__main__":
    # Quick test
    from src.data_fetcher import fetch_and_save, load_candles
    import os
    
    csv_path = os.path.join(os.path.dirname(__file__), "data", "btc_candles.csv")
    if os.path.exists(csv_path):
        df = load_candles()
    else:
        df = fetch_and_save(days=2)
    
    df = compute_features(df)
    X, y, feature_cols = prepare_ml_data(df)
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_cols}")
    print(f"Class balance: {y.value_counts().to_dict()}")
    print(df.tail(3))
