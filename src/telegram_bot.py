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


def _bar(value: float, width: int = 12) -> str:
    """Unicode block bar for visual meters (0.0 to 1.0)."""
    filled = round(value * width)
    empty = width - filled
    return "\u2593" * filled + "\u2591" * empty


def _pnl_display(amount: float) -> str:
    """Format P&L with sign prefix."""
    if amount >= 0:
        return f"+${amount:.2f}"
    return f"-${abs(amount):.2f}"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.40:
        return "MEDIUM"
    return "LOW"


def _streak_icon(streak_type: str) -> str:
    if streak_type and streak_type.lower() == "win":
        return "\u25b2"  # triangle up
    elif streak_type and streak_type.lower() == "loss":
        return "\u25bc"  # triangle down
    return "\u2014"  # em dash


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
    # Confidence: how far the probability is from 50/50 (0=coin flip, 1=certain)
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
    ISO strings for storage; display strings for messages.
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
    Full signal card — HTML formatted.
    Clear direction header, time window, confidence meter,
    key indicators, threshold info, and session stats footer.
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
        dir_icon = "\u25b2"  # triangle up
        dir_word = "UP"
    else:
        dir_icon = "\u25bc"  # triangle down
        dir_word = "DOWN"

    # ── Header
    lines = [
        f"<b>{dir_icon} BTC/USDT  {dir_word}</b>",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ]

    # ── Slot window
    lines += [
        f"<b>Slot</b>       {slot_open_disp} \u2192 {slot_close_disp}",
        f"<b>Price</b>      ${price:,.2f}",
        f"<b>Closes in</b>  {mins}m {secs}s",
        "",
    ]

    # ── Confidence
    lines += [
        f"<b>Confidence</b>",
        f"  {conf_bar}  {conf:.0%}  <b>{conf_label}</b>",
        f"  UP {prob_up:.1%}  \u00b7  DOWN {1 - prob_up:.1%}",
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
        lines.append("<b>Indicators</b>")
        lines.append("<code>" + "\n".join(indicator_lines) + "</code>")
        lines.append("")

    # ── Threshold
    lines.append(f"<b>Threshold</b>  {threshold:.3f}  <i>({thr_source})</i>")
    if model_wr:
        lines.append(f"<b>Model WR</b>   {model_wr:.1%}")

    # ── Session stats footer
    if stats and stats.get("total", 0) > 0:
        wr  = stats["win_rate"]
        pnl = stats["total_profit"]
        sk  = stats["current_streak"]
        skt = (stats["current_streak_type"] or "")
        sk_icon = _streak_icon(skt)

        lines += [
            "",
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"<b>Session</b>  {stats['total']} trades  \u00b7  WR {wr:.1%}  \u00b7  {_pnl_display(pnl)}",
            f"<b>Streak</b>   {sk_icon} {sk} {skt.upper()}",
        ]

    return "\n".join(lines)


def format_threshold_status(metrics: dict) -> str:
    """
    /threshold command response -- full breakdown of threshold sources.
    """
    threshold, source = resolve_threshold(metrics)
    trained   = metrics.get("threshold", PREDICTION_THRESHOLD)
    override  = get_runtime_threshold()
    model_wr  = metrics.get("validation_win_rate", 0)

    # Active marker
    def _mark(src_name):
        return "\u25c9" if source == src_name else "\u25cb"  # filled vs empty circle

    lines = [
        "<b>\u2699 Threshold Settings</b>",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"<b>Active</b>  {threshold:.3f}",
        "",
        "<b>Source Hierarchy</b>",
    ]

    # Runtime override
    if override is not None:
        lines.append(f"  {_mark('runtime override')}  Runtime override    {override:.3f}")
    else:
        lines.append(f"  {_mark('runtime override')}  Runtime override    <i>not set</i>")

    lines.append(f"  {_mark('model trained')}  Model trained       {trained:.3f}")
    lines.append(f"  {_mark('config default')}  Config default      {PREDICTION_THRESHOLD:.3f}")

    lines += [
        "",
        "<b>Impact Guide</b>",
        "<code>"
        f"  Lower ({THRESHOLD_MIN:.2f}+)   more signals, lower accuracy\n"
        f"  Higher (0.65+)  fewer signals, higher accuracy\n"
        f"  Optimal         {trained:.3f} (from backtest)"
        "</code>",
        "",
        "<b>Guidance</b>",
        f"  Paper trading    use <code>{trained:.3f}</code> (model default)",
        f"  Real money       raise to <code>0.60-0.70</code> for best edge",
    ]

    if model_wr:
        lines += [
            "",
            f"<b>Backtest WR</b>  {model_wr:.1%}  <i>(at trained threshold)</i>",
        ]

    lines += [
        "",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "Set:   <code>/setthreshold 0.62</code>",
        "Reset: <code>/setthreshold reset</code>",
    ]
    return "\n".join(lines)


def format_status_message(metrics: dict) -> str:
    """Model health dashboard for /status."""
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

    if val_acc >= 0.52:
        health = "\u25cf HEALTHY"
    else:
        health = "\u25cf BELOW BREAKEVEN"

    lines = [
        "<b>\u2699 Model Status</b>",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"<b>Health</b>  {health}",
        "",
        "<b>Training Info</b>",
        f"<code>"
        f"  Trained          {model_date}\n"
        f"  Training set     {n_train:,} samples\n"
        f"  Validation set   {n_val:,} trades"
        f"</code>",
        "",
        "<b>Performance</b>",
        f"<code>"
        f"  Accuracy         {val_acc:.2%}\n"
        f"  AUC-ROC          {val_auc:.4f}\n"
        f"  Win Rate         {win_rate:.2%}\n"
        f"  EV per $1        {ev:+.4f}"
        f"</code>",
        "",
        "<b>Threshold</b>",
        f"  Active    {threshold:.3f}  <i>({source})</i>",
        f"  Breakeven {BREAKEVEN_WIN_RATE:.1%}",
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

    # By confidence tier
    high_wr, high_n = _wr([t for t in resolved_trades if t.get("confidence", 0) >= 0.70])
    med_wr,  med_n  = _wr([t for t in resolved_trades if 0.40 <= t.get("confidence", 0) < 0.70])
    low_wr,  low_n  = _wr([t for t in resolved_trades if t.get("confidence", 0) < 0.40])

    # By direction
    up_wr,   up_n   = _wr([t for t in resolved_trades if t["direction_code"] == 1])
    down_wr, down_n = _wr([t for t in resolved_trades if t["direction_code"] == 0])

    threshold, source = resolve_threshold(metrics)
    model_wr = metrics.get("validation_win_rate", 0)

    if wr >= BREAKEVEN_WIN_RATE:
        verdict = "\u25cf PROFITABLE"
    else:
        verdict = "\u25cf BELOW BREAKEVEN"

    # Win rate bar
    wr_bar = _bar(wr)

    lines = [
        "<b>\u2593 Live Accuracy Report</b>",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"  {wr_bar}  <b>{wr:.1%}</b>  Win Rate",
        f"  <b>{_pnl_display(pnl)}</b>  Total P&amp;L",
        "",
        f"<b>Summary</b>",
        f"<code>"
        f"  Resolved       {total} trades\n"
        f"  Wins           {wins}\n"
        f"  Losses         {losses}\n"
        f"  EV per trade   {ev_pt:+.4f}"
        f"</code>",
        "",
        "<b>By Confidence Tier</b>",
        "<code>"
        f"  HIGH   (&gt;=70%)   {high_wr:>5.1%}   {high_n:>3} trades\n"
        f"  MEDIUM (40-70%)  {med_wr:>5.1%}   {med_n:>3} trades\n"
        f"  LOW    (&lt;40%)    {low_wr:>5.1%}   {low_n:>3} trades"
        "</code>",
        "",
        "<b>By Direction</b>",
        "<code>"
        f"  UP signals       {up_wr:>5.1%}   {up_n:>3} trades\n"
        f"  DOWN signals     {down_wr:>5.1%}   {down_n:>3} trades"
        "</code>",
        "",
        "<b>Model Validation</b>",
        "<code>"
        f"  Backtest WR      {model_wr:.1%}\n"
        f"  Accuracy         {metrics.get('validation_accuracy', 0):.2%}\n"
        f"  AUC              {metrics.get('validation_auc', 0):.4f}\n"
        f"  Threshold        {threshold:.3f}  ({source})\n"
        f"  Breakeven        {BREAKEVEN_WIN_RATE:.1%}"
        "</code>",
        "",
        f"<b>{verdict}</b>",
    ]
    return "\n".join(lines)


def format_start_message() -> str:
    """Welcome card for /start."""
    return "\n".join([
        "<b>NeoXG</b>  \u2502  BTC 5-Min Signal Bot",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        "XGBoost ML model predicting BTC/USDT",
        "5-min candle direction. Signals fire",
        "<b>15 seconds before</b> each new candle.",
        "",
        "<b>\u2500 Trading</b>",
        "  /signal         Live prediction now",
        "  /stats          Session stats + recent trades",
        "",
        "<b>\u2500 Analysis</b>",
        "  /accuracy       Detailed accuracy breakdown",
        "  /status         Model health + performance",
        "",
        "<b>\u2500 Settings</b>",
        "  /threshold      View threshold settings",
        "  /setthreshold   Set confidence threshold",
        "",
        "<b>\u2500 General</b>",
        "  /help           Command reference",
        "",
        "<i>Auto-signals run every 5 min continuously.</i>",
    ])


def format_help_message() -> str:
    """Detailed command reference for /help."""
    return "\n".join([
        "<b>\u2263 Command Reference</b>",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        "<b>/signal</b>",
        "  Get a live BTC 5-min prediction right now.",
        "  Includes direction, confidence, and indicators.",
        "",
        "<b>/stats</b>",
        "  Win rate, P&amp;L, streaks, and last 5 trades.",
        "  Resolves any pending trades first.",
        "",
        "<b>/accuracy</b>",
        "  Full breakdown by confidence tier and",
        "  direction. Live vs model validation.",
        "",
        "<b>/status</b>",
        "  Model accuracy, AUC, training date,",
        "  and current threshold configuration.",
        "",
        "<b>/threshold</b>",
        "  View active threshold, source hierarchy,",
        "  and trading guidance.",
        "",
        "<b>/setthreshold</b> <code>&lt;value&gt;</code>",
        f"  Set confidence threshold ({THRESHOLD_MIN} - {THRESHOLD_MAX}).",
        "  Example: <code>/setthreshold 0.62</code>",
        "  Reset:   <code>/setthreshold reset</code>",
        "",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "<i>Auto-signals fire 15s before every 5-min candle.</i>",
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
        """Register bot menu commands with Telegram (the / command picker)."""
        commands = [
            BotCommand("signal",       "Get live BTC 5-min prediction"),
            BotCommand("stats",        "Win rate, P&L, recent trades"),
            BotCommand("accuracy",     "Detailed accuracy breakdown"),
            BotCommand("status",       "Model health & performance"),
            BotCommand("threshold",    "View threshold settings"),
            BotCommand("setthreshold", "Set confidence threshold"),
            BotCommand("help",         "Command reference"),
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
                "<i>Fetching signal...</i>", parse_mode=ParseMode.HTML
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
                f"\u26a0 <b>Signal Error</b>\n\n<code>{_html_escape(str(e))}</code>\n\n"
                f"<i>Try again in a moment or check /status.</i>",
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
                f"\u26a0 <b>Stats Error</b>\n\n<code>{_html_escape(str(e))}</code>",
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
                f"\u26a0 <b>Status Error</b>\n\n<code>{_html_escape(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )

    async def accuracy_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            resolve_pending_trades()
            tracker_data = load_tracker()
            resolved = [t for t in tracker_data.get("trades", []) if t.get("resolved")]
            if not resolved:
                await update.message.reply_text(
                    "<b>\u2593 Live Accuracy Report</b>\n"
                    f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                    "No resolved trades yet.\n\n"
                    "<i>Auto-signals run every 5 min.\n"
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
                f"\u26a0 <b>Accuracy Error</b>\n\n<code>{_html_escape(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )

    async def threshold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current threshold settings and guidance."""
        try:
            _, metrics = load_model()
            await update.message.reply_text(
                format_threshold_status(metrics), parse_mode=ParseMode.HTML
            )
        except Exception as e:
            log.exception("threshold_cmd error")
            await update.message.reply_text(
                f"\u26a0 <b>Threshold Error</b>\n\n<code>{_html_escape(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )

    async def setthreshold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        /setthreshold 0.62   - set override
        /setthreshold reset  - clear override, revert to model trained
        """
        try:
            args = ctx.args
            if not args:
                await update.message.reply_text(
                    "<b>\u2699 Set Threshold</b>\n"
                    f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                    "<b>Usage:</b>\n"
                    "  <code>/setthreshold 0.62</code>    set threshold\n"
                    "  <code>/setthreshold reset</code>   revert to default\n\n"
                    f"Valid range: <code>{THRESHOLD_MIN}</code> - <code>{THRESHOLD_MAX}</code>",
                    parse_mode=ParseMode.HTML,
                )
                return

            raw = args[0].strip().lower()

            # Reset case
            if raw == "reset":
                clear_runtime_threshold()
                _, metrics = load_model()
                trained = metrics.get("threshold", PREDICTION_THRESHOLD)
                await update.message.reply_text(
                    "<b>\u2713 Threshold Reset</b>\n\n"
                    f"Reverted to model default.\n"
                    f"Active threshold: <code>{trained:.3f}</code>  <i>(model trained)</i>",
                    parse_mode=ParseMode.HTML,
                )
                return

            # Parse numeric value
            try:
                value = float(raw)
            except ValueError:
                await update.message.reply_text(
                    f"\u26a0 <b>Invalid Value</b>\n\n"
                    f"<code>{_html_escape(raw)}</code> is not a valid number.\n\n"
                    f"Use a number between <code>{THRESHOLD_MIN}</code> and <code>{THRESHOLD_MAX}</code>, "
                    f"or <code>reset</code>.",
                    parse_mode=ParseMode.HTML,
                )
                return

            # Validate range
            if not (THRESHOLD_MIN <= value <= THRESHOLD_MAX):
                await update.message.reply_text(
                    f"\u26a0 <b>Out of Range</b>\n\n"
                    f"<code>{value:.3f}</code> is outside the valid range.\n"
                    f"Valid: <code>{THRESHOLD_MIN}</code> - <code>{THRESHOLD_MAX}</code>\n\n"
                    "<b>Guidance:</b>\n"
                    "  Paper trading   <code>0.52 - 0.58</code>\n"
                    "  Real money      <code>0.60 - 0.70</code>",
                    parse_mode=ParseMode.HTML,
                )
                return

            # Warn if straying far from trained value
            _, metrics = load_model()
            trained = metrics.get("threshold", PREDICTION_THRESHOLD)
            set_runtime_threshold(value)

            diff = value - trained

            impact = ""
            if diff > 0.08:
                impact = (
                    "\n\n<i>\u26a0 Significantly above model optimal. "
                    "Fewer signals but not guaranteed more accurate.</i>"
                )
            elif diff < -0.08:
                impact = (
                    "\n\n<i>\u26a0 Significantly below model optimal. "
                    "More signals but lower expected accuracy.</i>"
                )

            await update.message.reply_text(
                "<b>\u2713 Threshold Updated</b>\n\n"
                f"<code>"
                f"  New threshold   {value:.3f}  (runtime override)\n"
                f"  Model trained   {trained:.3f}\n"
                f"  Difference      {diff:+.3f}"
                f"</code>"
                + impact,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.exception("setthreshold_cmd error")
            await update.message.reply_text(
                f"\u26a0 <b>Threshold Error</b>\n\n<code>{_html_escape(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )

    async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "\u26a0 Unknown command.\n\nType /help for the full list.",
            parse_mode=ParseMode.HTML,
        )

    # ── Candle-aligned auto-signal ─────────────────────────────────────────────

    async def _auto_signal_job(ctx: ContextTypes.DEFAULT_TYPE):
        """Fires 15s before candle boundary. Resolves old trades, sends signal, reschedules."""
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

    # ── Post-init: register menu commands ────────────────────────────────────

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
