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
            print("✅ Data and model already exist — skipping provisioning")
        return False

    if verbose:
        print("🔧 First-time setup — provisioning data and model...")
        print("   This will take ~2-3 minutes on first run")
        print("   (90 days of candles + XGBoost training)")
        print()

    # ── Step 1: Fetch candles ────────────────────────────────────────────────
    if not candles_ok:
        if verbose:
            print("📡 Fetching 90 days of BTC 5-min candles from MEXC...")
        from src.data_fetcher import fetch_and_save
        t0 = time.time()
        try:
            fetch_and_save(days=90)
            if verbose:
                print(f"   ✅ Candles saved in {time.time()-t0:.0f}s")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Fetch failed: {e}")
            raise
    else:
        if verbose:
            print(f"✅ Candles already exist ({candles_path.stat().st_size//1024}KB)")

    # ── Step 2: Train model ─────────────────────────────────────────────────
    if not model_ok:
        if verbose:
            print("🤖 Training XGBoost model (this takes ~1-2 minutes)...")
        from src.trainer import run_training
        t0 = time.time()
        try:
            # Use 60 days train, 30 val — faster than full 150/30
            model, metrics = run_training(days_train=60, days_val=30)
            if verbose:
                print(f"   ✅ Model trained in {time.time()-t0:.0f}s")
                print(f"   Validation accuracy: {metrics.get('validation_accuracy', 0):.2%}")
                print(f"   Win rate: {metrics.get('validation_win_rate', 0):.2%}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Training failed: {e}")
            raise
    else:
        if verbose:
            print(f"✅ Model already exists")

    if verbose:
        print()
        print("✅ Provisioning complete — starting bot...")

    return True


if __name__ == "__main__":
    provision()
