"""
data_fetcher.py
Fetches 5-min BTC/USDT candles from MEXC using ccxt.
No auth needed for public market data.
"""

import time
import pandas as pd
import ccxt
from pathlib import Path
from src.config import DATA_DIR


def fetch_historical_candles(symbol="BTC/USDT", interval="5m", days=90):
    """
    Fetch `days` worth of 5-min candles from MEXC.
    Uses `since` parameter to paginate backwards from current time.
    
    Returns:
        pandas DataFrame with OHLCV data, sorted ascending by timestamp
    """
    exchange = ccxt.mexc({"rateLimit": 100})
    
    # Calculate how many candles we need and the start timestamp
    ms_per_candle = 5 * 60 * 1000  # 5-min
    total_ms = days * 24 * 60 * 60 * 1000
    start_ts = int(time.time() * 1000) - total_ms
    
    print(f"Fetching ~{days} days from MEXC ({symbol}, {interval})...")
    print(f"  Start: {pd.to_datetime(start_ts, unit='ms')}")
    
    all_candles = []
    current_ts = start_ts
    iteration = 0
    
    while True:
        iteration += 1
        batch = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=interval,
            since=current_ts,
            limit=1000
        )
        
        if not batch:
            break
        
        all_candles.extend(batch)
        
        # Move timestamp forward to avoid re-fetching the same candle
        last_ts = batch[-1][0]
        
        # If we've reached current time, stop
        now_ms = int(time.time() * 1000)
        if last_ts >= now_ms - ms_per_candle:
            # Cap to last complete candle
            print(f"  Reached current time at {len(all_candles)} candles")
            break
        
        current_ts = last_ts + 1
        
        if iteration % 10 == 0:
            print(f"  Iter {iteration}: {len(all_candles)} candles fetched, "
                  f"up to {pd.to_datetime(last_ts, unit='ms').strftime('%Y-%m-%d %H:%M')}")
        
        time.sleep(exchange.rateLimit / 1000)
    
    if not all_candles:
        raise Exception("No candles returned from MEXC!")
    
    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Deduplicate (same timestamp → keep last)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    
    expected = days * 24 * 60 // 5
    coverage = len(df) / expected * 100 if expected > 0 else 0
    
    print(f"  ✅ Total: {len(df)} candles")
    print(f"  Expected ~{expected}, coverage: {coverage:.1f}%")
    print(f"  Range: {df.datetime.min().strftime('%Y-%m-%d %H:%M')} → {df.datetime.max().strftime('%Y-%m-%d %H:%M')}")
    
    return df


def fetch_live_candles(symbol="BTC/USDT", interval="5m", lookback=200):
    """
    Fetch the most recent `lookback` closed candles for live prediction.
    """
    exchange = ccxt.mexc({"rateLimit": 100})
    
    candles = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=interval,
        limit=lookback
    )
    
    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df = df.tail(lookback).reset_index(drop=True)
    
    return df


def save_candles(df, filename="btc_candles.csv"):
    """Save candles to CSV."""
    path = Path(DATA_DIR) / filename
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} candles to {path}")


def load_candles(filename="btc_candles.csv"):
    """Load candles from CSV."""
    path = Path(DATA_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"No candle data at {path}. Run fetch first.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def fetch_and_save(days=90, filename="btc_candles.csv"):
    """Fetch historical candles and save to CSV."""
    df = fetch_historical_candles(days=days)
    save_candles(df, filename)
    return df


if __name__ == "__main__":
    # Quick test: fetch 2 days
    df = fetch_historical_candles(days=2)
    print(df.tail(5))
