"""
provision.py
Auto-provisions data and model on first startup.
Checks if data/model exists, fetches candles and trains model if missing.
"""

import os
import sys
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, MODEL_PATH, LOGS_DIR

log = logging.getLogger(__name__)


def check_healthy():
    """Check if data and model files exist."""
    candles_path = DATA_DIR / "btc_candles.csv"
    model_path = MODEL_PATH

    candles_ok = candles_path.exists() and candles_path.stat().st_size > 100_000
    model_ok = model_path.exists()

    return candles_ok and model_ok


def provision(verbose=True):
    """
    Fetch candles and train model if either is missing.
    Returns True if provisioning was needed, False if already set up.
    """
    candles_path = DATA_DIR / "btc_candles.csv"
    model_path = MODEL_PATH

    candles_ok = candles_path.exists() and candles_path.stat().st_size > 100_000
    model_ok = model_path.exists()

    if candles_ok and model_ok:
        if verbose:
            print("\u2705 Data and model already exist \u2014 skipping provisioning")
        return False

    if verbose:
        print("\U0001f527 First-time setup \u2014 provisioning data and model...")
        print("   This will take ~3-4 minutes on first run")
        print("   (200 days of candles + XGBoost/LightGBM training)")
        print()

    # \u2500\u2500 Step 1: Fetch candles \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if not candles_ok:
        if verbose:
            print("\U0001f4e1 Fetching 200 days of BTC 5-min candles from MEXC...")
        from src.data_fetcher import fetch_and_save
        t0 = time.time()
        try:
            fetch_and_save(days=200)
            if verbose:
                print(f"   \u2705 Candles saved in {time.time()-t0:.0f}s")
        except Exception as e:
            if verbose:
                print(f"   \u26a0\ufe0f  Fetch failed: {e}")
            raise
    else:
        if verbose:
            print(f"\u2705 Candles already exist ({candles_path.stat().st_size//1024}KB)")

    # \u2500\u2500 Step 2: Train model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if not model_ok:
        if verbose:
            print("\U0001f916 Training XGBoost + LightGBM ensemble (this takes ~2-3 minutes)...")
        from src.trainer import run_training
        t0 = time.time()
        try:
            # Use 120 days train, 30 val \u2014 good balance of accuracy vs speed
            model, metrics = run_training(days_train=120, days_val=30)
            if verbose:
                print(f"   \u2705 Model trained in {time.time()-t0:.0f}s")
                print(f"   Validation accuracy: {metrics.get('validation_accuracy', 0):.2%}")
                print(f"   Win rate: {metrics.get('validation_win_rate', 0):.2%}")
        except Exception as e:
            if verbose:
                print(f"   \u26a0\ufe0f  Training failed: {e}")
            raise
    else:
        if verbose:
            print(f"\u2705 Model already exists")

    if verbose:
        print()
        print("\u2705 Provisioning complete \u2014 starting bot...")

    return True


if __name__ == "__main__":
    provision()
