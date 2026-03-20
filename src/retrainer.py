"""
retrainer.py
Automated model retraining with champion-challenger validation.

Features:
  - Signal-safe lock: prevents retrain from overlapping with signal generation
  - Champion-challenger: only promotes new model if it outperforms the current one
  - Consecutive rejection tracking: force-accepts after N rejections to avoid staleness
  - Fresh data fetch before each retrain
  - Full Telegram notification support
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    MODEL_PATH, MODEL_BACKUP_PATH, DATA_DIR, LOGS_DIR, MODELS_DIR,
    RETRAIN_MAX_CONSECUTIVE_REJECTIONS,
)

log = logging.getLogger(__name__)

# ── Signal-safe retrain lock ────────────────────────────────────────────────
# The lock prevents retraining from running while a signal is being generated
# and vice versa. Signals acquire the lock briefly (~2-5s), retrains hold it
# for ~1-2 minutes. Non-blocking check ensures signals are never delayed.

_retrain_lock = threading.Lock()
_retrain_in_progress = False


def is_retrain_in_progress() -> bool:
    """Check if a retrain is currently running (non-blocking)."""
    return _retrain_in_progress


def acquire_signal_lock(timeout: float = 5.0) -> bool:
    """
    Acquire lock for signal generation. Returns True if acquired.
    If retrain is in progress, waits up to `timeout` seconds.
    Signals should always succeed — retrain is the one that yields.
    """
    return _retrain_lock.acquire(timeout=timeout)


def release_signal_lock() -> None:
    """Release signal generation lock."""
    try:
        _retrain_lock.release()
    except RuntimeError:
        pass  # already released


# ── Retrain state persistence ───────────────────────────────────────────────

RETRAIN_STATE_FILE = os.path.join(DATA_DIR, "retrain_state.json")


def _load_retrain_state() -> dict:
    """Load retrain state from disk."""
    if os.path.exists(RETRAIN_STATE_FILE):
        try:
            with open(RETRAIN_STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            log.warning("Retrain state file corrupted — resetting.")
    return {
        "last_retrain_utc": None,
        "total_retrains": 0,
        "total_upgrades": 0,
        "total_rejections": 0,
        "consecutive_rejections": 0,
        "last_result": None,
        "history": [],
    }


def _save_retrain_state(state: dict) -> None:
    """Save retrain state atomically."""
    tmp = RETRAIN_STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RETRAIN_STATE_FILE)


def get_retrain_state() -> dict:
    """Public accessor for retrain state."""
    return _load_retrain_state()


# ── Champion-challenger comparison ──────────────────────────────────────────

def _compare_models(old_metrics: dict, new_metrics: dict) -> dict:
    """
    Compare old (champion) vs new (challenger) model metrics.

    Returns dict with:
      - upgrade: bool — True if new model should replace old
      - reason: str — human-readable explanation
      - old_wr / new_wr: win rates
      - old_acc / new_acc: accuracies
      - forced: bool — True if upgrade was forced due to consecutive rejections
    """
    old_wr = old_metrics.get("validation_win_rate", 0)
    new_wr = new_metrics.get("validation_win_rate", 0)
    old_acc = old_metrics.get("validation_accuracy", 0)
    new_acc = new_metrics.get("validation_accuracy", 0)
    old_ev = old_metrics.get("expected_value_per_dollar", 0)
    new_ev = new_metrics.get("expected_value_per_dollar", 0)

    state = _load_retrain_state()
    consec_rejections = state.get("consecutive_rejections", 0)

    result = {
        "old_wr": old_wr,
        "new_wr": new_wr,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "old_ev": old_ev,
        "new_ev": new_ev,
        "forced": False,
    }

    # Primary comparison: win rate (most relevant for profitability)
    # Secondary: accuracy as tiebreaker
    if new_wr > old_wr:
        result["upgrade"] = True
        result["reason"] = (
            f"New model wins: WR {new_wr:.2%} > {old_wr:.2%} "
            f"(+{(new_wr - old_wr):.2%})"
        )
    elif new_wr == old_wr and new_acc > old_acc:
        result["upgrade"] = True
        result["reason"] = (
            f"Same WR ({new_wr:.2%}), better accuracy: "
            f"{new_acc:.2%} > {old_acc:.2%}"
        )
    elif consec_rejections + 1 >= RETRAIN_MAX_CONSECUTIVE_REJECTIONS:
        # Force-accept: old model is likely stale
        result["upgrade"] = True
        result["forced"] = True
        result["reason"] = (
            f"FORCED UPGRADE after {consec_rejections + 1} consecutive rejections. "
            f"Old model likely stale. "
            f"WR: {old_wr:.2%} -> {new_wr:.2%}"
        )
    else:
        result["upgrade"] = False
        result["reason"] = (
            f"Old model kept: WR {old_wr:.2%} >= {new_wr:.2%} "
            f"(rejection {consec_rejections + 1}/{RETRAIN_MAX_CONSECUTIVE_REJECTIONS})"
        )

    return result


# ── Core retrain pipeline ───────────────────────────────────────────────────

def run_retrain(force_accept: bool = False) -> dict:
    """
    Full retrain pipeline with champion-challenger validation.

    Steps:
      1. Acquire retrain lock (waits for any active signal to finish)
      2. Fetch fresh candle data
      3. Train new model to a temporary path
      4. Compare new vs current model metrics
      5. Promote or reject the new model
      6. Update retrain state
      7. Release lock

    Args:
        force_accept: If True, skip comparison and always accept new model.

    Returns:
        dict with retrain results (upgrade, metrics, reason, etc.)
    """
    global _retrain_in_progress

    # ── Acquire lock ────────────────────────────────────────────────────
    log.info("Retrain: waiting for lock...")
    acquired = _retrain_lock.acquire(timeout=120)  # wait up to 2 min
    if not acquired:
        log.error("Retrain: could not acquire lock after 120s — aborting.")
        return {
            "success": False,
            "error": "Could not acquire retrain lock (signal generation may be stuck).",
        }

    _retrain_in_progress = True
    start_time = time.time()

    try:
        log.info("Retrain: lock acquired, starting pipeline...")

        # ── Load current model metrics (champion) ──────────────────────
        metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
        old_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                old_metrics = json.load(f)

        # ── Fetch fresh data ───────────────────────────────────────────
        log.info("Retrain: fetching fresh candle data...")
        from src.data_fetcher import fetch_and_save
        fetch_and_save(days=150)
        log.info("Retrain: fresh data fetched.")

        # ── Train new model to temp path ───────────────────────────────
        log.info("Retrain: training new model...")
        temp_model_path = MODELS_DIR / "btc_direction_model_challenger.json"
        temp_metrics_path = MODELS_DIR / "btc_direction_model_challenger_metrics.json"

        from src.trainer import run_training
        new_model, new_metrics = run_training(days_train=120, days_val=30)

        # Save challenger to temp path
        new_model.save_model(str(temp_model_path))
        with open(str(temp_metrics_path), "w") as f:
            json.dump(new_metrics, f, indent=2, default=str)

        log.info(
            f"Retrain: challenger trained — "
            f"WR={new_metrics.get('validation_win_rate', 0):.2%}, "
            f"Acc={new_metrics.get('validation_accuracy', 0):.2%}"
        )

        # ── Compare champion vs challenger ─────────────────────────────
        if force_accept:
            comparison = {
                "upgrade": True,
                "forced": True,
                "reason": "Force-accepted by user request.",
                "old_wr": old_metrics.get("validation_win_rate", 0),
                "new_wr": new_metrics.get("validation_win_rate", 0),
                "old_acc": old_metrics.get("validation_accuracy", 0),
                "new_acc": new_metrics.get("validation_accuracy", 0),
                "old_ev": old_metrics.get("expected_value_per_dollar", 0),
                "new_ev": new_metrics.get("expected_value_per_dollar", 0),
            }
        else:
            comparison = _compare_models(old_metrics, new_metrics)

        # ── Promote or reject ──────────────────────────────────────────
        state = _load_retrain_state()

        if comparison["upgrade"]:
            # Backup current model before replacing
            if os.path.exists(MODEL_PATH):
                shutil.copy2(str(MODEL_PATH), str(MODEL_BACKUP_PATH))
                backup_metrics = str(MODEL_BACKUP_PATH).replace(".json", "_metrics.json")
                if os.path.exists(metrics_path):
                    shutil.copy2(metrics_path, backup_metrics)
                log.info("Retrain: current model backed up.")

            # Promote challenger to champion
            shutil.move(str(temp_model_path), str(MODEL_PATH))
            shutil.move(str(temp_metrics_path), metrics_path)
            log.info("Retrain: new model PROMOTED to champion.")

            state["total_upgrades"] += 1
            state["consecutive_rejections"] = 0
            state["last_result"] = "upgraded"
        else:
            # Reject challenger — clean up temp files
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            if os.path.exists(temp_metrics_path):
                os.remove(temp_metrics_path)
            log.info("Retrain: challenger REJECTED, keeping current model.")

            state["total_rejections"] += 1
            state["consecutive_rejections"] += 1
            state["last_result"] = "rejected"

        # ── Update state ───────────────────────────────────────────────
        elapsed = round(time.time() - start_time, 1)
        state["last_retrain_utc"] = datetime.now(timezone.utc).isoformat()
        state["total_retrains"] += 1

        # Keep last 20 entries in history
        state["history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": "upgraded" if comparison["upgrade"] else "rejected",
            "forced": comparison.get("forced", False),
            "old_wr": comparison["old_wr"],
            "new_wr": comparison["new_wr"],
            "old_acc": comparison["old_acc"],
            "new_acc": comparison["new_acc"],
            "elapsed_seconds": elapsed,
            "reason": comparison["reason"],
        })
        state["history"] = state["history"][-20:]

        _save_retrain_state(state)

        return {
            "success": True,
            "upgrade": comparison["upgrade"],
            "forced": comparison.get("forced", False),
            "reason": comparison["reason"],
            "old_wr": comparison["old_wr"],
            "new_wr": comparison["new_wr"],
            "old_acc": comparison["old_acc"],
            "new_acc": comparison["new_acc"],
            "old_ev": comparison.get("old_ev", 0),
            "new_ev": comparison.get("new_ev", 0),
            "elapsed_seconds": elapsed,
            "consecutive_rejections": state["consecutive_rejections"],
        }

    except Exception as e:
        log.exception("Retrain: pipeline failed")
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        _retrain_in_progress = False
        try:
            _retrain_lock.release()
        except RuntimeError:
            pass
        log.info("Retrain: lock released.")
