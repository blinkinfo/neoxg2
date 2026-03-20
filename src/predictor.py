"""
predictor.py
Live prediction script -- runs every 5 mins.
Fetches latest candles + microstructure data from MEXC,
computes features, predicts direction using XGBoost + LightGBM ensemble.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.config import (
    MODEL_PATH, LIGHTGBM_MODEL_PATH, DATA_DIR, LOGS_DIR,
    PREDICTION_THRESHOLD, MEXC_SYMBOL, MEXC_INTERVAL,
    ORDERBOOK_DEPTH, FUNDING_RATE_SYMBOL,
    ENSEMBLE_WEIGHTS, MIN_CONFIDENCE_TO_TRADE,
    HIGH_CONFIDENCE_THRESHOLD,
    VOLATILITY_REGIME_LOOKBACK, LOW_VOLATILITY_ATR_PERCENTILE,
)
from src.data_fetcher import (
    fetch_live_candles, fetch_order_book_imbalance,
    fetch_funding_rate, fetch_open_interest_mexc,
)
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


def load_lgb_model():
    """Load trained LightGBM model. Returns None if not available."""
    try:
        import lightgbm as lgb
        if not os.path.exists(LIGHTGBM_MODEL_PATH):
            return None
        model = lgb.Booster(model_file=str(LIGHTGBM_MODEL_PATH))
        return model
    except Exception as e:
        print(f"LightGBM model load failed: {e}")
        return None


def _get_model_feature_names(xgb_model, lgb_booster):
    """
    Extract the feature names the models were actually trained on.
    Uses XGBoost as the source of truth (always available).
    Returns list of feature names.
    """
    try:
        xgb_features = xgb_model.get_booster().feature_names
        if xgb_features:
            return xgb_features
    except Exception:
        pass

    # Fallback: try LightGBM
    if lgb_booster is not None:
        try:
            lgb_features = lgb_booster.feature_name()
            if lgb_features:
                return lgb_features
        except Exception:
            pass

    # Last resort: return None (no filtering will be applied)
    return None


def ensemble_predict(xgb_model, lgb_booster, X, feature_cols, weights=None):
    """
    Ensemble prediction from XGBoost + LightGBM.
    Falls back to XGBoost-only if LightGBM not available.
    Returns probability of UP (class 1).
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    xgb_proba = xgb_model.predict_proba(X)[:, 1]

    if lgb_booster is not None:
        try:
            lgb_proba = lgb_booster.predict(X)
            proba = weights["xgboost"] * xgb_proba + weights["lightgbm"] * lgb_proba
        except Exception as e:
            print(f"LightGBM prediction failed, using XGBoost only: {e}")
            proba = xgb_proba
    else:
        proba = xgb_proba

    return proba


def predict_direction(model, df, threshold=None, lgb_booster=None):
    """
    Given a DataFrame of candles (with latest candle at index -1),
    compute features, add microstructure data, and predict direction.
    Uses ensemble of XGBoost + LightGBM if available.

    Returns dict with prediction info.
    """
    if threshold is None:
        try:
            metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
            with open(metrics_path) as f:
                metrics = json.load(f)
            threshold = metrics.get("threshold", PREDICTION_THRESHOLD)
        except Exception:
            threshold = PREDICTION_THRESHOLD

    # Compute features
    df_feat = compute_features(df.copy())
    
    # Fetch microstructure data and add to last row
    ob_data = fetch_order_book_imbalance(MEXC_SYMBOL, ORDERBOOK_DEPTH)
    fr_data = fetch_funding_rate(FUNDING_RATE_SYMBOL)
    oi_data = fetch_open_interest_mexc("BTC_USDT")
    
    # Add microstructure features to the dataframe
    last_idx = df_feat.index[-1]
    for key, val in ob_data.items():
        df_feat.loc[last_idx, key] = val
    df_feat.loc[last_idx, "funding_rate"] = fr_data["funding_rate"]
    
    # Open interest change (current vs previous -- need historical for %)
    # For now, store raw OI; the pct change would need historical context
    df_feat.loc[last_idx, "open_interest"] = oi_data["open_interest"]
    
    # Prepare features
    X, y, feature_cols = prepare_ml_data(df_feat, drop_na=False)
    
    # Get the last valid row (may have NaNs in microstructure cols -- that's OK)
    X_last = X.iloc[[-1]]
    
    # Fill any NaN in microstructure cols with 0 (neutral signal)
    X_last = X_last.fillna(0)

    # --- FIX: Align inference features to model's trained features ---
    # The model was trained on OHLCV-derived features only (30 features).
    # Microstructure features (bid_ask_imbalance, top5_imbalance, spread_pct,
    # funding_rate) are injected at inference but don't exist in the trained
    # model. We must filter X_last to only the features the model expects.
    trained_features = _get_model_feature_names(model, lgb_booster)
    if trained_features is not None:
        # Only keep columns the model was trained on, in the correct order
        missing = [f for f in trained_features if f not in X_last.columns]
        if missing:
            # Shouldn't happen, but guard against it: fill missing with 0
            for col in missing:
                X_last[col] = 0
        X_last = X_last[trained_features]
        feature_cols = list(trained_features)

    # Ensemble predict
    proba = ensemble_predict(model, lgb_booster, X_last, feature_cols)
    proba = float(proba[0])
    
    prediction = 1 if proba >= threshold else 0
    confidence = abs(proba - 0.5) * 2  # 0 = 50/50, 1 = 100% confident

    # Direction
    direction = "UP" if prediction == 1 else "DOWN"

    # Volatility regime check
    last_candle = df_feat.iloc[-1]
    is_low_vol = bool(last_candle.get("low_volatility", 0))
    
    # Confidence filtering
    skip_trade = False
    skip_reason = None
    if confidence < MIN_CONFIDENCE_TO_TRADE:
        skip_trade = True
        skip_reason = f"Confidence {confidence:.1%} below minimum {MIN_CONFIDENCE_TO_TRADE:.1%}"
    elif is_low_vol:
        skip_trade = True
        skip_reason = "Low volatility regime -- signal unreliable"

    # Confidence tier
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_tier = "HIGH"
    elif confidence >= 0.20:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "LOW"

    # Most recent candle info
    prev_close = df_feat.iloc[-2]["close"]
    candle_change_pct = (last_candle["close"] - prev_close) / prev_close * 100

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": direction,
        "direction_code": prediction,
        "probability_up": round(proba, 4),
        "confidence": round(confidence, 4),
        "confidence_tier": confidence_tier,
        "threshold_used": threshold,
        "last_candle_close": round(float(last_candle["close"]), 2),
        "last_candle_change_pct": round(candle_change_pct, 4),
        "rsi": round(float(last_candle["rsi"]), 2) if not np.isnan(last_candle.get("rsi", np.nan)) else None,
        "macd_histogram": round(float(last_candle["histogram"]), 4) if not np.isnan(last_candle.get("histogram", np.nan)) else None,
        "volume_ratio": round(float(last_candle["volume_ratio"]), 2) if not np.isnan(last_candle.get("volume_ratio", np.nan)) else None,
        "bb_position": round(float(last_candle.get("bb_position", np.nan)), 2) if not np.isnan(last_candle.get("bb_position", np.nan)) else None,
        "atr_pct": round(float(last_candle.get("atr_pct", np.nan)), 4) if not np.isnan(last_candle.get("atr_pct", np.nan)) else None,
        # New microstructure indicators
        "bid_ask_imbalance": ob_data.get("bid_ask_imbalance"),
        "top5_imbalance": ob_data.get("top5_imbalance"),
        "spread_pct": ob_data.get("spread_pct"),
        "funding_rate": fr_data.get("funding_rate"),
        "is_low_volatility": is_low_vol,
        "skip_trade": skip_trade,
        "skip_reason": skip_reason,
        "ensemble": lgb_booster is not None,
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

    emoji = "\U0001f7e2" if result["direction_code"] == 1 else "\U0001f534"

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
    """Main prediction runner -- called every 5 mins."""
    log_file = os.path.join(LOGS_DIR, f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl")

    print(f"\n[{datetime.utcnow().isoformat()}] Running prediction...")

    try:
        model, metrics = load_model()
        lgb_booster = load_lgb_model()

        df = fetch_live_candles(
            symbol=MEXC_SYMBOL,
            interval=MEXC_INTERVAL,
            lookback=200
        )
        print(f"  Fetched {len(df)} candles: {df.datetime.min().strftime('%H:%M')} -> {df.datetime.max().strftime('%H:%M')}")

        result, df_feat = predict_direction(model, df, lgb_booster=lgb_booster)

        msg = format_signal_message(result)
        print(f"  {msg.replace(chr(10), ' | ')}")

        with open(log_file, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        return result

    except Exception as e:
        error_msg = f"Prediction error: {e}"
        print(f"  {error_msg}")
        with open(log_file, "a") as f:
            f.write(json.dumps({"error": str(e), "timestamp": datetime.utcnow().isoformat()}) + "\n")
        raise


if __name__ == "__main__":
    run_prediction()
