"""
predictor.py
Live prediction script — runs every 5 mins.
Fetches latest candles from MEXC, computes features, predicts direction, logs result.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.config import (
    MODEL_PATH, DATA_DIR, LOGS_DIR, PREDICTION_THRESHOLD,
    MEXC_SYMBOL, MEXC_INTERVAL
)
from src.data_fetcher import fetch_live_candles
from src.features import compute_features, prepare_ml_data


def load_model():
    """Load trained XGBoost model."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Load metrics
    metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return model, metrics


def predict_direction(model, df, threshold=None):
    """
    Given a DataFrame of candles (with latest candle at index -1),
    compute features and predict the direction of the NEXT candle.

    Returns dict with prediction info.
    """
    if threshold is None:
        # Load from saved metrics
        try:
            metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
            with open(metrics_path) as f:
                metrics = json.load(f)
            threshold = metrics.get("threshold", PREDICTION_THRESHOLD)
        except Exception:
            threshold = PREDICTION_THRESHOLD

    # Compute features
    df_feat = compute_features(df.copy())
    X, y, feature_cols = prepare_ml_data(df_feat, drop_na=True)

    # Predict on last row (last complete candle)
    X_last = X.iloc[[-1]]
    proba = model.predict_proba(X_last)[0, 1]
    prediction = 1 if proba >= threshold else 0
    confidence = abs(proba - 0.5) * 2  # 0 = 50/50, 1 = 100% confident

    # Direction
    direction = "UP" if prediction == 1 else "DOWN"

    # Most recent candle info
    last_candle = df_feat.iloc[-1]
    last_close = last_candle["close"]
    prev_close = df_feat.iloc[-2]["close"]
    candle_change_pct = (last_close - prev_close) / prev_close * 100

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": direction,
        "direction_code": prediction,
        "probability_up": round(proba, 4),
        "confidence": round(confidence, 4),
        "threshold_used": threshold,
        "last_candle_close": round(last_close, 2),
        "last_candle_change_pct": round(candle_change_pct, 4),
        "rsi": round(last_candle["rsi"], 2) if not np.isnan(last_candle["rsi"]) else None,
        "macd_histogram": round(last_candle["histogram"], 4) if not np.isnan(last_candle["histogram"]) else None,
        "volume_ratio": round(last_candle["volume_ratio"], 2) if not np.isnan(last_candle["volume_ratio"]) else None,
    }

    return result, df_feat


def format_signal_message(result):
    """Format prediction result as a clean Telegram message."""
    prob = result["probability_up"]
    conf = result["confidence"]
    rsi = result["rsi"]
    vol = result["volume_ratio"]
    price = result["last_candle_close"]
    change = result["last_candle_change_pct"]

    # Emoji based on direction
    emoji = "\U0001f7e2" if result["direction_code"] == 1 else "\U0001f534"

    # Signal strength
    if conf >= 0.7:
        strength = "HIGH CONFIDENCE"
    elif conf >= 0.4:
        strength = "MEDIUM"
    else:
        strength = "LOW CONFIDENCE"

    msg = (
        f"{emoji} BTC 5m Signal\n"
        f"Direction: {result['prediction']}\n"
        f"Probability UP: {prob:.1%}\n"
        f"Confidence: {conf:.1%}\n"
        f"Last Close: ${price:,.1f}\n"
        f"Change: {change:+.3f}%\n"
        f"RSI: {rsi}\n"
        f"Vol Ratio: {vol}x\n"
        f"{strength}\n"
        f"Trade: $1 {result['prediction']}\n"
        f"Time: {result['timestamp']}"
    )
    return msg


def run_prediction():
    """Main prediction runner — called every 5 mins."""
    log_file = os.path.join(LOGS_DIR, f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl")

    print(f"\n[{datetime.utcnow().isoformat()}] Running prediction...")

    try:
        # Load model
        model, metrics = load_model()

        # Get latest candles from MEXC
        df = fetch_live_candles(
            symbol=MEXC_SYMBOL,
            interval=MEXC_INTERVAL,
            lookback=200
        )
        print(f"  Fetched {len(df)} candles: {df.datetime.min().strftime('%H:%M')} -> {df.datetime.max().strftime('%H:%M')}")

        # Predict
        result, df_feat = predict_direction(model, df)

        # Format and print
        msg = format_signal_message(result)
        print(f"  {msg.replace(chr(10), ' | ')}")

        # Log to file
        with open(log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result

    except Exception as e:
        error_msg = f"Prediction error: {e}"
        print(f"  {error_msg}")
        with open(log_file, "a") as f:
            f.write(json.dumps({"error": str(e), "timestamp": datetime.utcnow().isoformat()}) + "\n")
        raise


if __name__ == "__main__":
    run_prediction()
