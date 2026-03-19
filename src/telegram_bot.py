"""
telegram_bot.py
BTC 5-min signal bot with candle-aligned auto-signals, runtime threshold
configuration, and full Polymarket-ready UX.

Signals fire 15 seconds BEFORE each 5-min candle boundary.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MODEL_PATH, DATA_DIR, LOGS_DIR, PREDICTION_THRESHOLD,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADE_AMOUNT, PAYOUT
)
from src.data_fetcher import fetch_live_candles
from src.features import compute_features, prepare_ml_data
from src.tracker import (
    record_signal, resolve_trade, get_stats,
    format_stats_message, format_recent_trades_message,
    load_tracker
)
from src.threshold import (
    resolve_threshold, set_runtime_threshold, clear_runtime_threshold,
    get_runtime_threshold, THRESHOLD_MIN, THRESHOLD_MAX
)

# ── Constants ───────────────────────────────────────────────────────────────────
SIGNAL_LEAD_SECONDS = 15
CANDLE_MINUTES = 5
BREAKEVEN_WIN_RATE = 1 / (1 + PAYOUT)   # ~51.0% for 0.96 payout

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "telegram_bot.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _html_escape(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bar(value: float, width: int = 10) -> str:
    """Green/gray square bar for visual meters (0.0 to 1.0)."""
    filled = round(value * width)
    empty = width - filled
    return "\U0001f7e9" * filled + "\u2b1c" * empty


def _pnl_emoji(amount: float) -> str:
    """P&L with emoji prefix."""
    if amount > 0:
        return f"\U0001f4b0 +${amount:.2f}"
    elif amount < 0:
        return f"\U0001f534 -${abs(amount):.2f}"
    return "\u26aa $0.00"


def _pnl_display(amount: float) -> str:
    """Format P&L with sign prefix (no emoji)."""
    if amount >= 0:
        return f"+${amount:.2f}"
    return f"-${abs(amount):.2f}"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "\U0001f525 HIGH"
    if confidence >= 0.40:
        return "\u26a1 MEDIUM"
    return "\U0001f4a8 LOW"


def _confidence_emoji(confidence: float) -> str:
    if confidence >= 0.70:
        return "\U0001f525"
    if confidence >= 0.40:
        return "\u26a1"
    return "\U0001f4a8"


def _streak_display(streak: int, streak_type: str) -> str:
    if not streak_type:
        return "\u2796 No streak"
    if streak_type.lower() == "win":
        return f"\U0001f525 {streak}W streak"
    return f"\U0001f9ca {streak}L streak"


def _verdict_line(win_rate: float) -> str:
    """Profitability verdict with emoji."""
    breakeven = 1 / (1 + PAYOUT)
    if win_rate >= breakeven + 0.05:
        return "\U0001f4b9 <b>PROFITABLE</b>"
    elif win_rate >= breakeven:
        return "\u2696\ufe0f <b>BREAKEVEN</b>"
    else:
        return "\U0001f4c9 <b>BELOW BREAKEVEN</b>"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL / PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load trained XGBoost model and its metrics."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, metrics


def get_live_prediction():
    """
    Fetch latest candles, run model, return prediction dict + metrics.
    Threshold is resolved in priority order:
      1. Runtime override (/setthreshold)
      2. Model trained threshold
      3. Config default
    """
    model, metrics = load_model()
    threshold, threshold_source = resolve_threshold(metrics)

    df = fetch_live_candles(lookback=200)
    df_feat = compute_features(df.copy())
    X, _, _ = prepare_ml_data(df_feat, drop_na=True)

    X_last = X.iloc[[-1]]
    proba = model.predict_proba(X_last)[0, 1]
    prediction = 1 if proba >= threshold else 0
    confidence = abs(proba - 0.5) * 2

    last_c = df_feat.iloc[-1]
    prev_c = df_feat.iloc[-2]

    def _safe(val):
        return None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    return {
        "prediction":      "UP" if prediction == 1 else "DOWN",
        "direction_code":  prediction,
        "probability_up":  round(proba, 4),
        "confidence":      round(confidence, 4),
        "threshold":       threshold,
        "threshold_source": threshold_source,
        "last_close":      round(last_c["close"], 2),
        "prev_close":      round(prev_c["close"], 2),
        "rsi":             _safe(round(last_c["rsi"], 2) if "rsi" in last_c else None),
        "macd_histogram":  _safe(round(last_c["histogram"], 4) if "histogram" in last_c else None),
        "volume_ratio":    _safe(round(last_c["volume_ratio"], 2) if "volume_ratio" in last_c else None),
        "bb_position":     _safe(round(last_c["bb_position"], 2) if "bb_position" in last_c else None),
        "atr_pct":         _safe(round(last_c["atr_pct"], 4) if "atr_pct" in last_c else None),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }, metrics


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE TIMING
# ══════════════════════════════════════════════════════════════════════════════

def _next_candle_boundary():
    """Next exact 5-min UTC boundary. E.g. 12:03:22 -> 12:05:00."""
    now = datetime.now(timezone.utc)
    base = now.replace(second=0, microsecond=0)
    remainder = base.minute % CANDLE_MINUTES
    if remainder == 0 and now.second == 0 and now.microsecond == 0:
        return base + timedelta(minutes=CANDLE_MINUTES)
    return base + timedelta(minutes=CANDLE_MINUTES - remainder)


def get_next_slot_times():
    """
    Returns (slot_open_iso, slot_close_iso, slot_open_disp, slot_close_disp,
             mins_until, secs_until)
    """
    slot_open_dt  = _next_candle_boundary()
    slot_close_dt = slot_open_dt + timedelta(minutes=CANDLE_MINUTES)
    now = datetime.now(timezone.utc)
    diff = (slot_open_dt - now).total_seconds()
    return (
        slot_open_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        slot_close_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        slot_open_dt.strftime("%H:%M UTC"),
        slot_close_dt.strftime("%H:%M UTC"),
        int(diff // 60),
        int(diff % 60),
    )


def _parse_slot_time(slot_str):
    """Parse ISO-8601 or legacy 'HH:MM UTC' to aware UTC datetime."""
    if "T" in slot_str:
        return datetime.strptime(slot_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t   = datetime.strptime(slot_str, "%H:%M UTC")
    now = datetime.now(timezone.utc)
    dt  = datetime.combine(now.date(), t.time(), tzinfo=timezone.utc)
    if (dt - now).total_seconds() >  12 * 3600:
        dt -= timedelta(days=1)
    elif (now - dt).total_seconds() > 12 * 3600:
        dt += timedelta(days=1)
    return dt


# ══════════════════════════════════════════════════════════════════════════════
# TRADE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_pending_trades():
    """Resolve all expired unresolved trades via exact candle timestamp match."""
    now = datetime.now(timezone.utc)
    tracker_data = load_tracker()
    expired = [
        t for t in tracker_data.get("trades", [])
        if not t.get("resolved") and now >= _parse_slot_time(t["slot_close"])
    ]
    if not expired:
        return

    live_df = fetch_live_candles(lookback=100)
    for trade in expired:
        slot_open_dt = _parse_slot_time(trade["slot_open"])
        open_ts_ms   = int(slot_open_dt.timestamp() * 1000)
        matched = live_df[live_df["timestamp"] == open_ts_ms]
        if matched.empty:
            tol = 60_000
            matched = live_df[
                (live_df["timestamp"] >= open_ts_ms - tol) &
                (live_df["timestamp"] <= open_ts_ms + tol)
            ]
        if matched.empty:
            log.warning(f"No candle for trade {trade['id']} slot={trade['slot_open']}")
            continue
        candle = matched.iloc[0]
        outcome = 1 if candle["close"] > candle["open"] else 0
        resolved = resolve_trade(trade["id"], outcome)
        if resolved:
            log.info(f"Resolved trade {trade['id']}: {resolved['result']}")


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE FORMATTING  (HTML parse mode)
# ══════════════════════════════════════════════════════════════════════════════

def format_signal_message(result: dict, metrics: dict, stats: dict) -> str:
    """
    Full signal card -- rich HTML with emojis.
    """
    _, _, slot_open_disp, slot_close_disp, mins, secs = get_next_slot_times()

    direction    = result["prediction"]
    conf         = result["confidence"]
    conf_label   = _confidence_label(conf)
    conf_bar     = _bar(conf)
    prob_up      = result["probability_up"]
    threshold    = result["threshold"]
    thr_source   = result["threshold_source"]
    model_wr     = metrics.get("validation_win_rate", 0)
    price        = result["last_close"]

    if direction == "UP":
        dir_emoji = "\U0001f7e2"   # green circle
        dir_label = "\U0001f53c UP"
    else:
        dir_emoji = "\U0001f534"   # red circle
        dir_label = "\U0001f53d DOWN"

    # ── Header
    lines = [
        f"{dir_emoji} <b>BTC/USDT  \u2502  {dir_label}</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ]

    # ── Slot + Price
    lines += [
        f"\u23f0  <b>Slot:</b>  {slot_open_disp} \u2192 {slot_close_disp}",
        f"\U0001f4b2  <b>Price:</b>  ${price:,.2f}",
        f"\u23f3  <b>Closes in:</b>  {mins}m {secs}s",
        "",
    ]

    # ── Confidence
    lines += [
        f"\U0001f3af  <b>Confidence:</b>  {conf:.0%}  {conf_label}",
        f"      {conf_bar}",
        f"      \U0001f53c UP {prob_up:.1%}   \u2502   \U0001f53d DOWN {1 - prob_up:.1%}",
        "",
    ]

    # ── Indicators
    rsi  = result.get("rsi")
    macd = result.get("macd_histogram")
    vol  = result.get("volume_ratio")
    bb   = result.get("bb_position")
    atr  = result.get("atr_pct")

    indicator_lines = []
    if rsi  is not None: indicator_lines.append(f"  RSI           {rsi}")
    if macd is not None: indicator_lines.append(f"  MACD Hist     {macd:+.4f}")
    if vol  is not None: indicator_lines.append(f"  Vol Ratio     {vol:.2f}x")
    if bb   is not None: indicator_lines.append(f"  BB Position   {bb:.2f}")
    if atr  is not None: indicator_lines.append(f"  ATR %         {atr:.3%}")

    if indicator_lines:
        lines.append("\U0001f4ca  <b>Indicators</b>")
        lines.append("<code>" + "\n".join(indicator_lines) + "</code>")
        lines.append("")

    # ── Threshold + Model
    lines.append(f"\u2699\ufe0f  <b>Threshold:</b>  {threshold:.3f}  <i>({thr_source})</i>")
    if model_wr:
        lines.append(f"\U0001f9e0  <b>Model WR:</b>  {model_wr:.1%}")

    # ── Session stats footer
    if stats and stats.get("total", 0) > 0:
        wr  = stats["win_rate"]
        pnl = stats["total_profit"]
        sk  = stats["current_streak"]
        skt = (stats["current_streak_type"] or "")

        lines += [
            "",
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"\U0001f4c8  <b>Session:</b>  {stats['total']} trades  \u2502  WR {wr:.1%}  \u2502  {_pnl_display(pnl)}",
            f"{_streak_display(sk, skt)}",
        ]

    return "\n".join(lines)


def format_threshold_status(metrics: dict) -> str:
    """/threshold command response."""
    threshold, source = resolve_threshold(metrics)
    trained   = metrics.get("threshold", PREDICTION_THRESHOLD)
    override  = get_runtime_threshold()
    model_wr  = metrics.get("validation_win_rate", 0)

    def _mark(src_name):
        return "\U0001f7e2" if source == src_name else "\u26aa"  # green vs gray circle

    lines = [
        "\u2699\ufe0f  <b>Threshold Settings</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"\U0001f3af  <b>Active:</b>  <code>{threshold:.3f}</code>",
        "",
        "\U0001f4da  <b>Source Hierarchy</b>",
    ]

    if override is not None:
        lines.append(f"      {_mark('runtime override')}  Runtime Override  \u2192  <code>{override:.3f}</code>")
    else:
        lines.append(f"      {_mark('runtime override')}  Runtime Override  \u2192  <i>not set</i>")

    lines.append(f"      {_mark('model trained')}  Model Trained  \u2192  <code>{trained:.3f}</code>")
    lines.append(f"      {_mark('config default')}  Config Default  \u2192  <code>{PREDICTION_THRESHOLD:.3f}</code>")

    lines += [
        "",
        "\U0001f4a1  <b>Impact Guide</b>",
        "<code>"
        f"  Lower ({THRESHOLD_MIN:.2f}+)   More signals, lower accuracy\n"
        f"  Higher (0.65+)  Fewer signals, higher accuracy\n"
        f"  Optimal         {trained:.3f} (from backtest)"
        "</code>",
        "",
        "\U0001f9ed  <b>Trading Guidance</b>",
        f"      \U0001f4dd  Paper trading  \u2192  <code>{trained:.3f}</code>  (model default)",
        f"      \U0001f4b5  Real money  \u2192  <code>0.60 - 0.70</code>  (best edge)",
    ]

    if model_wr:
        lines += [
            "",
            f"\U0001f9e0  <b>Backtest WR:</b>  {model_wr:.1%}  <i>(at trained threshold)</i>",
        ]

    lines += [
        "",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "\u270d\ufe0f  Set:  <code>/setthreshold 0.62</code>",
        "\U0001f504  Reset:  <code>/setthreshold reset</code>",
    ]
    return "\n".join(lines)


def format_status_message(metrics: dict) -> str:
    """Model health dashboard for /status with profitability verdict."""
    threshold, source = resolve_threshold(metrics)
    val_acc  = metrics.get("validation_accuracy", 0)
    val_auc  = metrics.get("validation_auc", 0)
    win_rate = metrics.get("validation_win_rate", 0)
    ev       = metrics.get("expected_value_per_dollar", 0)
    n_val    = metrics.get("total_validation_trades", 0)
    n_train  = metrics.get("training_samples", 0)

    model_date = "Unknown"
    if os.path.exists(MODEL_PATH):
        model_date = datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).strftime("%Y-%m-%d %H:%M UTC")

    # Health badge
    if val_acc >= 0.55:
        health = "\U0001f7e2 EXCELLENT"
    elif val_acc >= 0.52:
        health = "\U0001f7e1 HEALTHY"
    else:
        health = "\U0001f534 WEAK"

    # Profitability verdict from model validation
    verdict = _verdict_line(win_rate)

    # EV emoji
    if ev > 0:
        ev_emoji = "\U0001f4b0"
    elif ev == 0:
        ev_emoji = "\u26aa"
    else:
        ev_emoji = "\U0001f534"

    lines = [
        "\U0001f916  <b>Model Status</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"\U0001f3e5  <b>Health:</b>  {health}",
        f"{verdict}",
        "",
        "\U0001f4da  <b>Training Info</b>",
        "<code>"
        f"  Trained          {model_date}\n"
        f"  Training set     {n_train:,} samples\n"
        f"  Validation set   {n_val:,} trades"
        "</code>",
        "",
        "\U0001f4ca  <b>Performance</b>",
        "<code>"
        f"  Accuracy         {val_acc:.2%}\n"
        f"  AUC-ROC          {val_auc:.4f}\n"
        f"  Win Rate         {win_rate:.2%}\n"
        f"  EV per $1        {ev:+.4f}"
        "</code>",
        "",
        f"{ev_emoji}  <b>Expected Value:</b>  {ev:+.4f} per $1 wagered",
        "",
        "\u2699\ufe0f  <b>Threshold</b>",
        f"      Active:  <code>{threshold:.3f}</code>  <i>({source})</i>",
        f"      Breakeven:  {BREAKEVEN_WIN_RATE:.1%}",
    ]
    return "\n".join(lines)


def format_accuracy_message(resolved_trades: list, metrics: dict) -> str:
    """Detailed live accuracy breakdown for /accuracy."""
    total  = len(resolved_trades)
    wins   = sum(1 for t in resolved_trades if t["result"] == "WIN")
    losses = total - wins
    wr     = wins / total if total else 0
    pnl    = sum(t.get("profit", 0) for t in resolved_trades)
    ev_pt  = pnl / total if total else 0

    def _wr(lst):
        if not lst: return 0.0, 0
        w = sum(1 for t in lst if t["result"] == "WIN")
        return w / len(lst), len(lst)

    high_wr, high_n = _wr([t for t in resolved_trades if t.get("confidence", 0) >= 0.70])
    med_wr,  med_n  = _wr([t for t in resolved_trades if 0.40 <= t.get("confidence", 0) < 0.70])
    low_wr,  low_n  = _wr([t for t in resolved_trades if t.get("confidence", 0) < 0.40])

    up_wr,   up_n   = _wr([t for t in resolved_trades if t["direction_code"] == 1])
    down_wr, down_n = _wr([t for t in resolved_trades if t["direction_code"] == 0])

    threshold, source = resolve_threshold(metrics)
    model_wr = metrics.get("validation_win_rate", 0)

    verdict = _verdict_line(wr)
    wr_bar  = _bar(wr)

    lines = [
        "\U0001f4cb  <b>Live Accuracy Report</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"      {wr_bar}  <b>{wr:.1%}</b>",
        f"      {_pnl_emoji(pnl)}  Total P&amp;L",
        f"      {verdict}",
        "",
        "\U0001f4ca  <b>Summary</b>",
        "<code>"
        f"  Resolved       {total} trades\n"
        f"  Wins           {wins}  \u2713\n"
        f"  Losses         {losses}  \u2717\n"
        f"  EV per trade   {ev_pt:+.4f}"
        "</code>",
        "",
        "\U0001f3af  <b>By Confidence Tier</b>",
        "<code>"
        f"  \U0001f525 HIGH   (&gt;=70%)   {high_wr:>5.1%}   {high_n:>3} trades\n"
        f"  \u26a1 MED    (40-70%)  {med_wr:>5.1%}   {med_n:>3} trades\n"
        f"  \U0001f4a8 LOW    (&lt;40%)    {low_wr:>5.1%}   {low_n:>3} trades"
        "</code>",
        "",
        "\U0001f504  <b>By Direction</b>",
        "<code>"
        f"  \U0001f53c UP signals       {up_wr:>5.1%}   {up_n:>3} trades\n"
        f"  \U0001f53d DOWN signals     {down_wr:>5.1%}   {down_n:>3} trades"
        "</code>",
        "",
        "\U0001f9e0  <b>Model Validation</b>",
        "<code>"
        f"  Backtest WR      {model_wr:.1%}\n"
        f"  Accuracy         {metrics.get('validation_accuracy', 0):.2%}\n"
        f"  AUC              {metrics.get('validation_auc', 0):.4f}\n"
        f"  Threshold        {threshold:.3f}  ({source})\n"
        f"  Breakeven        {BREAKEVEN_WIN_RATE:.1%}"
        "</code>",
    ]
    return "\n".join(lines)


def format_start_message() -> str:
    """Welcome card for /start."""
    return "\n".join([
        "\U0001f680  <b>NeoXG</b>  \u2502  BTC 5-Min Signal Bot",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        "XGBoost ML model predicting BTC/USDT",
        "5-min candle direction. Signals fire",
        "<b>15 seconds before</b> each new candle.",
        "",
        "\U0001f4c8  <b>Trading</b>",
        "      /signal  \u2014  Live prediction now",
        "      /stats  \u2014  Session stats + recent trades",
        "",
        "\U0001f50d  <b>Analysis</b>",
        "      /accuracy  \u2014  Detailed accuracy breakdown",
        "      /status  \u2014  Model health + performance",
        "",
        "\u2699\ufe0f  <b>Settings</b>",
        "      /threshold  \u2014  View threshold settings",
        "      /setthreshold  \u2014  Set confidence threshold",
        "",
        "\u2753  <b>General</b>",
        "      /help  \u2014  Command reference",
        "",
        "\U0001f552  <i>Auto-signals run every 5 min continuously.</i>",
    ])


def format_help_message() -> str:
    """Detailed command reference for /help."""
    return "\n".join([
        "\U0001f4d6  <b>Command Reference</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        "\U0001f7e2  <b>/signal</b>",
        "      Get a live BTC 5-min prediction right now.",
        "      Includes direction, confidence, and indicators.",
        "",
        "\U0001f4ca  <b>/stats</b>",
        "      Win rate, P&amp;L, streaks, and last 5 trades.",
        "      Resolves any pending trades first.",
        "",
        "\U0001f4cb  <b>/accuracy</b>",
        "      Full breakdown by confidence tier and",
        "      direction. Live vs model validation.",
        "",
        "\U0001f916  <b>/status</b>",
        "      Model accuracy, AUC, training date,",
        "      and current threshold configuration.",
        "",
        "\u2699\ufe0f  <b>/threshold</b>",
        "      View active threshold, source hierarchy,",
        "      and trading guidance.",
        "",
        "\u270d\ufe0f  <b>/setthreshold</b> <code>&lt;value&gt;</code>",
        f"      Set confidence threshold ({THRESHOLD_MIN} - {THRESHOLD_MAX}).",
        "      Example: <code>/setthreshold 0.62</code>",
        "      Reset:  <code>/setthreshold reset</code>",
        "",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "\U0001f552  <i>Auto-signals fire 15s before every 5-min candle.</i>",
    ])


# ══════════════════════════════════════════════════════════════════════════════
# BOT
# ══════════════════════════════════════════════════════════════════════════════

def run_bot():
    from telegram import Update, BotCommand
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )
    from telegram.constants import ParseMode

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")
        return

    from src.provision import provision, check_healthy
    if not check_healthy():
        log.info("Data/model missing - provisioning on startup...")
        provision(verbose=False)
    else:
        log.info("Data and model present - starting bot.")

    # ── Menu Commands Setup ───────────────────────────────────────────────────

    async def _set_menu_commands(app_instance):
        """Register bot menu commands with Telegram."""
        commands = [
            BotCommand("signal",       "\U0001f7e2 Get live BTC prediction"),
            BotCommand("stats",        "\U0001f4ca Win rate, P&L, trades"),
            BotCommand("accuracy",     "\U0001f4cb Accuracy breakdown"),
            BotCommand("status",       "\U0001f916 Model health & perf"),
            BotCommand("threshold",    "\u2699\ufe0f Threshold settings"),
            BotCommand("setthreshold", "\u270d\ufe0f Set threshold value"),
            BotCommand("help",         "\U0001f4d6 Command reference"),
        ]
        await app_instance.bot.set_my_commands(commands)
        log.info("Menu commands registered with Telegram.")

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            format_start_message(), parse_mode=ParseMode.HTML
        )

    async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            format_help_message(), parse_mode=ParseMode.HTML
        )

    async def signal_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text(
                "\u23f3 <i>Fetching signal...</i>", parse_mode=ParseMode.HTML
            )
            resolve_pending_trades()
            pred, metrics = get_live_prediction()
            stats = get_stats()
            slot_open_iso, slot_close_iso, _, _, _, _ = get_next_slot_times()

            trade = record_signal(
                slot_open_time=slot_open_iso,
                slot_close_time=slot_close_iso,
                direction=pred["prediction"],
                direction_code=pred["direction_code"],
                probability_up=pred["probability_up"],
                confidence=pred["confidence"],
                close_at_signal=pred["last_close"],
                rsi=pred["rsi"],
                macd_histogram=pred["macd_histogram"],
                volume_ratio=pred["volume_ratio"],
            )
            log.info(f"Signal #{trade['id']}: {pred['prediction']} prob={pred['probability_up']:.4f}")
            await update.message.reply_text(
                format_signal_message(pred, metrics, stats),
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.exception("signal_cmd error")
            await update.message.reply_text(
                f"\u26a0\ufe0f <b>Signal Error</b>\n\n"
                f"<code>{_html_escape(str(e))}</code>\n\n"
                f"\U0001f504 <i>Try again in a moment or check /status.</i>",
                parse_mode=ParseMode.HTML,
            )

    async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            resolve_pending_trades()
            await update.message.reply_text(
                format_stats_message(), parse_mode=ParseMode.HTML
            )
            recent = format_recent_trades_message(5)
            if recent:
                await update.message.reply_text(
                    recent, parse_mode=ParseMode.HTML
                )
        except Exception as e:
            log.exception("stats_cmd error")
            await update.message.reply_text(
                f"\u26a0\ufe0f <b>Stats Error</b>\n\n"
                f"<code>{_html_escape(str(e))}</code>\n\n"
                f"\U0001f504 <i>Try again shortly.</i>",
                parse_mode=ParseMode.HTML,
            )

    async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            _, metrics = load_model()
            await update.message.reply_text(
                format_status_message(metrics), parse_mode=ParseMode.HTML
            )
        except Exception as e:
            log.exception("status_cmd error")
            await update.message.reply_text(
                f"\u26a0\ufe0f <b>Status Error</b>\n\n"
                f"<code>{_html_escape(str(e))}</code>\n\n"
                f"\U0001f504 <i>Try again shortly.</i>",
                parse_mode=ParseMode.HTML,
            )

    async def accuracy_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            resolve_pending_trades()
            tracker_data = load_tracker()
            resolved = [t for t in tracker_data.get("trades", []) if t.get("resolved")]
            if not resolved:
                await update.message.reply_text(
                    "\U0001f4cb  <b>Live Accuracy Report</b>\n"
                    "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                    "\U0001f4ed  No resolved trades yet.\n\n"
                    "\U0001f552  <i>Auto-signals run every 5 min.\n"
                    "Trades resolve when their candle closes.\n"
                    "Check back in a few minutes!</i>",
                    parse_mode=ParseMode.HTML,
                )
                return
            _, metrics = load_model()
            await update.message.reply_text(
                format_accuracy_message(resolved, metrics),
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.exception("accuracy_cmd error")
            await update.message.reply_text(
                f"\u26a0\ufe0f <b>Accuracy Error</b>\n\n"
                f"<code>{_html_escape(str(e))}</code>\n\n"
                f"\U0001f504 <i>Try again shortly.</i>",
                parse_mode=ParseMode.HTML,
            )

    async def threshold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            _, metrics = load_model()
            await update.message.reply_text(
                format_threshold_status(metrics), parse_mode=ParseMode.HTML
            )
        except Exception as e:
            log.exception("threshold_cmd error")
            await update.message.reply_text(
                f"\u26a0\ufe0f <b>Threshold Error</b>\n\n"
                f"<code>{_html_escape(str(e))}</code>\n\n"
                f"\U0001f504 <i>Try again shortly.</i>",
                parse_mode=ParseMode.HTML,
            )

    async def setthreshold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            args = ctx.args
            if not args:
                await update.message.reply_text(
                    "\u270d\ufe0f  <b>Set Threshold</b>\n"
                    "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                    "\U0001f4dd  <b>Usage:</b>\n"
                    "      <code>/setthreshold 0.62</code>  \u2014  set threshold\n"
                    "      <code>/setthreshold reset</code>  \u2014  revert to default\n\n"
                    f"\U0001f4cf  Valid range:  <code>{THRESHOLD_MIN}</code> \u2014 <code>{THRESHOLD_MAX}</code>",
                    parse_mode=ParseMode.HTML,
                )
                return

            raw = args[0].strip().lower()

            if raw == "reset":
                clear_runtime_threshold()
                _, metrics = load_model()
                trained = metrics.get("threshold", PREDICTION_THRESHOLD)
                await update.message.reply_text(
                    "\u2705  <b>Threshold Reset</b>\n\n"
                    f"Reverted to model default.\n"
                    f"Active threshold:  <code>{trained:.3f}</code>  <i>(model trained)</i>",
                    parse_mode=ParseMode.HTML,
                )
                return

            try:
                value = float(raw)
            except ValueError:
                await update.message.reply_text(
                    f"\u26a0\ufe0f  <b>Invalid Value</b>\n\n"
                    f"<code>{_html_escape(raw)}</code> is not a valid number.\n\n"
                    f"\U0001f4cf  Use a number between <code>{THRESHOLD_MIN}</code> and <code>{THRESHOLD_MAX}</code>, "
                    f"or <code>reset</code>.",
                    parse_mode=ParseMode.HTML,
                )
                return

            if not (THRESHOLD_MIN <= value <= THRESHOLD_MAX):
                await update.message.reply_text(
                    f"\u26a0\ufe0f  <b>Out of Range</b>\n\n"
                    f"<code>{value:.3f}</code> is outside the valid range.\n"
                    f"\U0001f4cf  Valid:  <code>{THRESHOLD_MIN}</code> \u2014 <code>{THRESHOLD_MAX}</code>\n\n"
                    "\U0001f4a1  <b>Guidance:</b>\n"
                    "      \U0001f4dd  Paper trading  \u2192  <code>0.52 - 0.58</code>\n"
                    "      \U0001f4b5  Real money  \u2192  <code>0.60 - 0.70</code>",
                    parse_mode=ParseMode.HTML,
                )
                return

            _, metrics = load_model()
            trained = metrics.get("threshold", PREDICTION_THRESHOLD)
            set_runtime_threshold(value)
            diff = value - trained

            impact = ""
            if diff > 0.08:
                impact = (
                    "\n\n\u26a0\ufe0f  <i>Significantly above model optimal. "
                    "Fewer signals but not guaranteed more accurate.</i>"
                )
            elif diff < -0.08:
                impact = (
                    "\n\n\u26a0\ufe0f  <i>Significantly below model optimal. "
                    "More signals but lower expected accuracy.</i>"
                )

            await update.message.reply_text(
                "\u2705  <b>Threshold Updated</b>\n\n"
                "<code>"
                f"  New threshold   {value:.3f}  (runtime override)\n"
                f"  Model trained   {trained:.3f}\n"
                f"  Difference      {diff:+.3f}"
                "</code>"
                + impact,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.exception("setthreshold_cmd error")
            await update.message.reply_text(
                f"\u26a0\ufe0f  <b>Threshold Error</b>\n\n"
                f"<code>{_html_escape(str(e))}</code>\n\n"
                f"\U0001f504 <i>Try again shortly.</i>",
                parse_mode=ParseMode.HTML,
            )

    async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "\u2753  Unknown command.\n\n"
            "\U0001f4d6  Type /help for the full command list.",
            parse_mode=ParseMode.HTML,
        )

    # ── Candle-aligned auto-signal ─────────────────────────────────────────────

    async def _auto_signal_job(ctx: ContextTypes.DEFAULT_TYPE):
        """Fires 15s before candle boundary."""
        try:
            resolve_pending_trades()
            pred, metrics = get_live_prediction()
            stats = get_stats()
            slot_open_iso, slot_close_iso, _, _, _, _ = get_next_slot_times()

            trade = record_signal(
                slot_open_time=slot_open_iso,
                slot_close_time=slot_close_iso,
                direction=pred["prediction"],
                direction_code=pred["direction_code"],
                probability_up=pred["probability_up"],
                confidence=pred["confidence"],
                close_at_signal=pred["last_close"],
                rsi=pred["rsi"],
                macd_histogram=pred["macd_histogram"],
                volume_ratio=pred["volume_ratio"],
            )
            await ctx.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=format_signal_message(pred, metrics, stats),
                parse_mode=ParseMode.HTML,
            )
            log.info(
                f"Auto-signal #{trade['id']}: {pred['prediction']} "
                f"prob={pred['probability_up']:.4f} slot={slot_open_iso}"
            )
        except Exception as e:
            log.exception("Auto-signal error")
        finally:
            _schedule_next_signal(ctx.job.data["app"])

    def _schedule_next_signal(app):
        now           = datetime.now(timezone.utc)
        next_boundary = _next_candle_boundary()
        fire_at       = next_boundary - timedelta(seconds=SIGNAL_LEAD_SECONDS)
        if fire_at <= now:
            next_boundary += timedelta(minutes=CANDLE_MINUTES)
            fire_at        = next_boundary - timedelta(seconds=SIGNAL_LEAD_SECONDS)
        delay = (fire_at - now).total_seconds()
        log.info(
            f"Next signal: {fire_at.strftime('%H:%M:%S UTC')} "
            f"(in {delay:.0f}s, slot {next_boundary.strftime('%H:%M UTC')})"
        )
        app.job_queue.run_once(
            _auto_signal_job,
            when=delay,
            data={"app": app},
        )

    # ── Post-init ─────────────────────────────────────────────────────────────

    async def _post_init(app_instance):
        await _set_menu_commands(app_instance)

    # ── Build app ────────────────────────────────────────────────────────────

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()

    app.add_handler(CommandHandler("start",        start_cmd))
    app.add_handler(CommandHandler("help",         help_cmd))
    app.add_handler(CommandHandler("signal",       signal_cmd))
    app.add_handler(CommandHandler("stats",        stats_cmd))
    app.add_handler(CommandHandler("status",       status_cmd))
    app.add_handler(CommandHandler("accuracy",     accuracy_cmd))
    app.add_handler(CommandHandler("threshold",    threshold_cmd))
    app.add_handler(CommandHandler("setthreshold", setthreshold_cmd))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))

    _schedule_next_signal(app)
    log.info("NeoXG bot started with candle-aligned signals.")
    app.run_polling()


if __name__ == "__main__":
    run_bot()
