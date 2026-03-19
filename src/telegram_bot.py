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
# MESSAGE FORMATTING  (full UX overhaul)
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.40:
        return "MEDIUM"
    return "LOW"


def _confidence_bar(confidence: float, width: int = 10) -> str:
    """ASCII progress bar for confidence (0-1)."""
    filled = round(confidence * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _pnl_emoji(amount: float) -> str:
    return "+" if amount >= 0 else ""


def format_signal_message(result: dict, metrics: dict, stats: dict) -> str:
    """
    Full signal card:
      - Clear direction header
      - Time window
      - Confidence meter
      - Key indicators
      - Threshold info (source shown)
      - Live session stats footer
    """
    _, _, slot_open_disp, slot_close_disp, mins, secs = get_next_slot_times()

    direction    = result["prediction"]       # "UP" or "DOWN"
    dir_emoji    = "\U0001f7e2" if direction == "UP" else "\U0001f534"  # green/red circle
    arrow        = "\U0001f4c8" if direction == "UP" else "\U0001f4c9"  # chart up/down
    conf         = result["confidence"]
    conf_label   = _confidence_label(conf)
    conf_bar     = _confidence_bar(conf)
    prob_up      = result["probability_up"]
    threshold    = result["threshold"]
    thr_source   = result["threshold_source"]
    model_wr     = metrics.get("validation_win_rate", 0)
    model_ev     = metrics.get("expected_value_per_dollar", 0)

    lines = [
        f"{dir_emoji} BTC/USDT  5m Signal",
        f"{arrow} {direction}   {slot_open_disp} -> {slot_close_disp}",
        "",
        f"Confidence   {conf_bar} {conf:.0%}  [{conf_label}]",
        f"Prob UP      {prob_up:.1%}    Prob DOWN  {1 - prob_up:.1%}",
        "",
        "---- Indicators ----",
    ]

    rsi  = result.get("rsi")
    macd = result.get("macd_histogram")
    vol  = result.get("volume_ratio")
    bb   = result.get("bb_position")
    atr  = result.get("atr_pct")

    if rsi  is not None: lines.append(f"RSI          {rsi}")
    if macd is not None: lines.append(f"MACD Hist    {macd:+.4f}")
    if vol  is not None: lines.append(f"Volume Ratio {vol:.2f}x")
    if bb   is not None: lines.append(f"BB Position  {bb:.2f}")
    if atr  is not None: lines.append(f"ATR %        {atr:.3%}")

    lines += [
        "",
        "---- Threshold ----",
        f"Active       {threshold:.3f}  ({thr_source})",
    ]
    if model_wr:
        lines.append(f"Model WR     {model_wr:.1%}   EV/$ {model_ev:+.4f}")

    lines += [
        "",
        f"Closes in    {mins}m {secs}s",
    ]

    # Session stats footer (only when we have trades)
    if stats and stats.get("total", 0) > 0:
        wr  = stats["win_rate"]
        pnl = stats["total_profit"]
        sk  = stats["current_streak"]
        skt = (stats["current_streak_type"] or "").upper()
        sk_emoji = "\U0001f7e2" if skt == "WIN" else "\U0001f534"
        lines += [
            "",
            "---- Session ----",
            f"Trades  {stats['total']}   WR {wr:.1%}   P&L {_pnl_emoji(pnl)}${pnl:.2f}",
            f"Streak  {sk_emoji} {sk} {skt}",
        ]

    return "\n".join(lines)


def format_threshold_status(metrics: dict) -> str:
    """
    /threshold command response — full breakdown of threshold sources
    and what impact changing it has.
    """
    threshold, source = resolve_threshold(metrics)
    trained   = metrics.get("threshold", PREDICTION_THRESHOLD)
    override  = get_runtime_threshold()
    model_wr  = metrics.get("validation_win_rate", 0)

    # Impact guide
    lines = [
        "\U0001f4ca  Threshold Settings",
        "",
        f"Active threshold   {threshold:.3f}  ({source})",
        f"Model trained      {trained:.3f}  (optimal from backtest)",
    ]
    if override is not None:
        lines.append(f"Your override      {override:.3f}  (set via /setthreshold)")
    else:
        lines.append(f"Your override      not set")

    lines += [
        "",
        "---- What threshold does ----",
        f"Signal fires when model confidence >= threshold.",
        f"  Lower ({THRESHOLD_MIN:.2f}+)  = more signals, lower accuracy",
        f"  Higher (0.65+) = fewer signals, higher accuracy",
        f"  Trained ({trained:.3f}) = statistically optimal",
        "",
        "---- Polymarket guidance ----",
        f"For paper trading   use {trained:.3f} (model default)",
        f"For real money      raise to 0.60-0.70 for best edge",
        "",
    ]
    if model_wr:
        lines.append(f"Model backtest WR  {model_wr:.1%}  (at trained threshold)")

    lines += [
        "",
        "To change:  /setthreshold 0.62",
        "To reset:   /setthreshold reset",
    ]
    return "\n".join(lines)


def format_status_message(metrics: dict) -> str:
    """Model health card for /status."""
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

    health = "\U00002705 Profitable" if val_acc >= 0.52 else "\U000026a0  Below breakeven"

    lines = [
        "\U0001f916  Model Status",
        "",
        f"Trained          {model_date}",
        f"Training samples {n_train:,}",
        f"Validation trades {n_val:,}",
        "",
        "---- Performance ----",
        f"Accuracy         {val_acc:.2%}",
        f"AUC-ROC          {val_auc:.4f}",
        f"Win Rate         {win_rate:.2%}",
        f"EV per $1        {ev:+.4f}",
        "",
        "---- Threshold ----",
        f"Active           {threshold:.3f}  ({source})",
        f"Breakeven WR     {BREAKEVEN_WIN_RATE:.1%}",
        "",
        health,
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

    verdict = "\U00002705 Profitable" if wr >= BREAKEVEN_WIN_RATE else "\U000026a0  Below breakeven"

    lines = [
        "\U0001f4c8  Live Accuracy Report",
        "",
        f"Resolved trades  {total}",
        f"Wins / Losses    {wins} / {losses}",
        f"Live Win Rate    {wr:.1%}",
        f"Total P&L        {_pnl_emoji(pnl)}${pnl:.2f}",
        f"EV per trade     {ev_pt:+.4f}",
        "",
        "---- By Confidence ----",
        f"HIGH  (>=70%)    {high_wr:.1%}  ({high_n} trades)",
        f"MEDIUM (40-70%)  {med_wr:.1%}  ({med_n} trades)",
        f"LOW   (<40%)     {low_wr:.1%}  ({low_n} trades)",
        "",
        "---- By Direction ----",
        f"UP signals       {up_wr:.1%}  ({up_n} trades)",
        f"DOWN signals     {down_wr:.1%}  ({down_n} trades)",
        "",
        "---- Model Validation ----",
        f"Backtest WR      {model_wr:.1%}",
        f"Accuracy         {metrics.get('validation_accuracy', 0):.2%}",
        f"AUC              {metrics.get('validation_auc', 0):.4f}",
        f"Threshold        {threshold:.3f}  ({source})",
        f"Breakeven        {BREAKEVEN_WIN_RATE:.1%}",
        "",
        verdict,
    ]
    return "\n".join(lines)


def format_start_message() -> str:
    return "\n".join([
        "\U0001f916  NeoXG  |  BTC 5-min Signal Bot",
        "",
        "XGBoost ML model predicting BTC/USDT 5-min candle direction.",
        "Signals fire 15s before each new candle — every 5 minutes.",
        "",
        "---- Commands ----",
        "/signal       Live prediction now",
        "/stats        Win/loss tracker + recent trades",
        "/status       Model health + performance",
        "/accuracy     Detailed live accuracy breakdown",
        "/threshold    View threshold settings",
        "/setthreshold Set confidence threshold",
        "/help         Command reference",
        "",
        "---- Threshold ----",
        "Controls minimum model confidence before a signal fires.",
        "Use /threshold to learn more, /setthreshold 0.62 to set.",
        "",
        "Auto-signals run continuously. Use /signal anytime.",
    ])


def format_help_message() -> str:
    return "\n".join([
        "\U0001f4cb  Command Reference",
        "",
        "/signal",
        "  Get a live BTC 5-min prediction right now.",
        "",
        "/stats",
        "  Win rate, P&L, streaks, and last 5 trades.",
        "",
        "/status",
        "  Model accuracy, AUC, training date, threshold.",
        "",
        "/accuracy",
        "  Full breakdown by confidence tier and direction.",
        "",
        "/threshold",
        "  View active threshold, model trained value,",
        "  and Polymarket guidance.",
        "",
        "/setthreshold <value>",
        "  Set confidence threshold (0.30 - 0.90).",
        "  Example: /setthreshold 0.62",
        "  Reset to model default: /setthreshold reset",
        "",
        "Auto-signals fire 15s before every 5-min candle.",
    ])


# ══════════════════════════════════════════════════════════════════════════════
# BOT
# ══════════════════════════════════════════════════════════════════════════════

def run_bot():
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")
        return

    from src.provision import provision, check_healthy
    if not check_healthy():
        log.info("Data/model missing - provisioning on startup...")
        provision(verbose=False)
    else:
        log.info("Data and model present - starting bot.")

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(format_start_message())

    async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(format_help_message())

    async def signal_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            msg = await update.message.reply_text("Fetching signal...")
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
            await update.message.reply_text(format_signal_message(pred, metrics, stats))
        except Exception as e:
            log.exception("signal_cmd error")
            await update.message.reply_text(f"Error generating signal: {e}")

    async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            resolve_pending_trades()
            await update.message.reply_text(format_stats_message())
            recent = format_recent_trades_message(5)
            if recent:
                await update.message.reply_text(recent)
        except Exception as e:
            log.exception("stats_cmd error")
            await update.message.reply_text(f"Error: {e}")

    async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            _, metrics = load_model()
            await update.message.reply_text(format_status_message(metrics))
        except Exception as e:
            log.exception("status_cmd error")
            await update.message.reply_text(f"Error: {e}")

    async def accuracy_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            resolve_pending_trades()
            tracker_data = load_tracker()
            resolved = [t for t in tracker_data.get("trades", []) if t.get("resolved")]
            if not resolved:
                await update.message.reply_text(
                    "No resolved trades yet.\n"
                    "Auto-signals run every 5 min. Check back soon!"
                )
                return
            _, metrics = load_model()
            await update.message.reply_text(format_accuracy_message(resolved, metrics))
        except Exception as e:
            log.exception("accuracy_cmd error")
            await update.message.reply_text(f"Error: {e}")

    async def threshold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current threshold settings and guidance."""
        try:
            _, metrics = load_model()
            await update.message.reply_text(format_threshold_status(metrics))
        except Exception as e:
            log.exception("threshold_cmd error")
            await update.message.reply_text(f"Error: {e}")

    async def setthreshold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        /setthreshold 0.62   - set override
        /setthreshold reset  - clear override, revert to model trained
        """
        try:
            args = ctx.args
            if not args:
                await update.message.reply_text(
                    "Usage:\n"
                    "  /setthreshold 0.62    set threshold\n"
                    "  /setthreshold reset   revert to model default\n"
                    f"\nValid range: {THRESHOLD_MIN} - {THRESHOLD_MAX}"
                )
                return

            raw = args[0].strip().lower()

            # Reset case
            if raw == "reset":
                clear_runtime_threshold()
                _, metrics = load_model()
                trained = metrics.get("threshold", PREDICTION_THRESHOLD)
                await update.message.reply_text(
                    "\U00002705  Threshold reset to model default.\n"
                    f"Active threshold: {trained:.3f}  (model trained)"
                )
                return

            # Parse numeric value
            try:
                value = float(raw)
            except ValueError:
                await update.message.reply_text(
                    f"Invalid value: '{raw}'\n"
                    f"Use a number between {THRESHOLD_MIN} and {THRESHOLD_MAX}, "
                    f"or 'reset'."
                )
                return

            # Validate range
            if not (THRESHOLD_MIN <= value <= THRESHOLD_MAX):
                await update.message.reply_text(
                    f"\U000026a0  Out of range: {value:.3f}\n"
                    f"Valid range: {THRESHOLD_MIN} - {THRESHOLD_MAX}\n\n"
                    "Guidance:\n"
                    "  Paper trading   0.52 - 0.58\n"
                    "  Real money      0.60 - 0.70"
                )
                return

            # Warn if straying far from trained value
            _, metrics = load_model()
            trained = metrics.get("threshold", PREDICTION_THRESHOLD)
            set_runtime_threshold(value)

            impact = ""
            diff = value - trained
            if diff > 0.08:
                impact = (
                    "\n\nNote: significantly above model optimal. "
                    "Fewer signals but not guaranteed more accurate."
                )
            elif diff < -0.08:
                impact = (
                    "\n\nNote: significantly below model optimal. "
                    "More signals but lower expected accuracy."
                )

            await update.message.reply_text(
                f"\U00002705  Threshold updated.\n\n"
                f"New threshold   {value:.3f}  (runtime override)\n"
                f"Model trained   {trained:.3f}\n"
                f"Difference      {diff:+.3f}"
                + impact
            )
        except Exception as e:
            log.exception("setthreshold_cmd error")
            await update.message.reply_text(f"Error: {e}")

    async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Unknown command. Type /help for the full list."
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

    # ── Build app ────────────────────────────────────────────────────────────

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

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
