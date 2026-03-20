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
    
    # Deduplicate (same timestamp -> keep last)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    
    expected = days * 24 * 60 // 5
    coverage = len(df) / expected * 100 if expected > 0 else 0
    
    print(f"  Total: {len(df)} candles")
    print(f"  Expected ~{expected}, coverage: {coverage:.1f}%")
    print(f"  Range: {df.datetime.min().strftime('%Y-%m-%d %H:%M')} -> {df.datetime.max().strftime('%Y-%m-%d %H:%M')}")
    
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


def fetch_order_book_imbalance(symbol="BTC/USDT", depth=20):
    """
    Fetch order book and compute bid/ask imbalance metrics.
    Returns dict with imbalance features, or dict of NaNs on failure.
    
    bid_ask_imbalance: (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    Range: -1 (all asks) to +1 (all bids). Positive = buying pressure.
    
    spread_pct: (best_ask - best_bid) / mid_price * 100
    
    top5_imbalance: same ratio but only top 5 levels (more sensitive to immediate pressure)
    """
    import numpy as np
    nan_result = {
        "bid_ask_imbalance": np.nan,
        "top5_imbalance": np.nan,
        "spread_pct": np.nan,
        "bid_depth": np.nan,
        "ask_depth": np.nan,
    }
    try:
        exchange = ccxt.mexc({"rateLimit": 100})
        ob = exchange.fetch_order_book(symbol, limit=depth)
        
        if not ob["bids"] or not ob["asks"]:
            return nan_result
        
        bids = ob["bids"]  # [[price, volume], ...]
        asks = ob["asks"]
        
        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)
        
        if total_bid_vol + total_ask_vol == 0:
            return nan_result
        
        bid_ask_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        
        # Top 5 levels imbalance (more reactive)
        top5_bid = sum(b[1] for b in bids[:5])
        top5_ask = sum(a[1] for a in asks[:5])
        top5_imbalance = (top5_bid - top5_ask) / (top5_bid + top5_ask) if (top5_bid + top5_ask) > 0 else 0
        
        # Spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid_price * 100 if mid_price > 0 else 0
        
        return {
            "bid_ask_imbalance": round(bid_ask_imbalance, 6),
            "top5_imbalance": round(top5_imbalance, 6),
            "spread_pct": round(spread_pct, 6),
            "bid_depth": round(total_bid_vol, 4),
            "ask_depth": round(total_ask_vol, 4),
        }
    except Exception as e:
        print(f"Order book fetch failed: {e}")
        return nan_result


def fetch_funding_rate(symbol="BTC/USDT:USDT"):
    """
    Fetch current funding rate from MEXC perpetual swap.
    Returns dict with funding_rate (float) or NaN on failure.
    
    Positive funding = longs pay shorts (bullish sentiment, potential reversal down)
    Negative funding = shorts pay longs (bearish sentiment, potential reversal up)
    """
    import numpy as np
    try:
        exchange = ccxt.mexc({"rateLimit": 100})
        # Load markets first for futures
        exchange.load_markets()
        result = exchange.fetch_funding_rate(symbol)
        
        funding_rate = result.get("fundingRate", None)
        if funding_rate is None:
            return {"funding_rate": np.nan}
        
        return {"funding_rate": round(float(funding_rate), 8)}
    except Exception as e:
        print(f"Funding rate fetch failed: {e}")
        return {"funding_rate": np.nan}


def fetch_open_interest_mexc(symbol="BTC_USDT"):
    """
    Fetch open interest from MEXC futures API directly (not via ccxt).
    Uses the contract ticker endpoint.
    Returns dict with open_interest value or NaN on failure.
    """
    import numpy as np
    import requests
    try:
        url = f"https://contract.mexc.com/api/v1/contract/ticker?symbol={symbol}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("success") and data.get("data"):
            ticker_data = data["data"]
            # holdVol = total open interest in contracts
            hold_vol = ticker_data.get("holdVol", None)
            if hold_vol is not None:
                return {"open_interest": float(hold_vol)}
        
        return {"open_interest": np.nan}
    except Exception as e:
        print(f"Open interest fetch failed: {e}")
        return {"open_interest": np.nan}


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
