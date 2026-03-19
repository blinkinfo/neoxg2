"""threshold.py — Runtime threshold management, persists across bot restarts.

Priority order (highest wins):
  1. Runtime override (set via /setthreshold)
  2. Model trained threshold (from model_metrics.json)
  3. PREDICTION_THRESHOLD from config.py (last resort fallback)
"""

import json
import logging
import os
from typing import Optional

from src.config import DATA_DIR, PREDICTION_THRESHOLD

log = logging.getLogger(__name__)

RUNTIME_CONFIG_FILE = os.path.join(DATA_DIR, "runtime_config.json")

THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.90


def _load_runtime_config() -> dict:
    if os.path.exists(RUNTIME_CONFIG_FILE):
        try:
            with open(RUNTIME_CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            log.warning("Runtime config corrupted - ignoring.")
    return {}


def _save_runtime_config(data: dict) -> None:
    tmp = RUNTIME_CONFIG_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, RUNTIME_CONFIG_FILE)


def get_runtime_threshold() -> Optional[float]:
    """Return user-set runtime override, or None if not set."""
    cfg = _load_runtime_config()
    val = cfg.get("threshold_override")
    return float(val) if val is not None else None


def set_runtime_threshold(value: float) -> None:
    """Persist a new runtime threshold. Raises ValueError if out of range."""
    if not (THRESHOLD_MIN <= value <= THRESHOLD_MAX):
        raise ValueError(
            f"Threshold must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}. Got {value:.3f}."
        )
    cfg = _load_runtime_config()
    cfg["threshold_override"] = round(value, 4)
    _save_runtime_config(cfg)
    log.info(f"Runtime threshold set to {value:.4f}")


def clear_runtime_threshold() -> None:
    """Remove runtime override, revert to model trained threshold."""
    cfg = _load_runtime_config()
    cfg.pop("threshold_override", None)
    _save_runtime_config(cfg)
    log.info("Runtime threshold override cleared.")


def resolve_threshold(model_metrics: dict) -> tuple:
    """
    Return (threshold_value, source_label).
    source_label: 'runtime override' | 'model trained' | 'config default'
    """
    override = get_runtime_threshold()
    if override is not None:
        return override, "runtime override"

    trained = model_metrics.get("threshold")
    if trained is not None:
        return float(trained), "model trained"

    return PREDICTION_THRESHOLD, "config default"
