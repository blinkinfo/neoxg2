"""
telegram_bot.py
Signal bot — sends BTC 5-min direction predictions to Telegram.
Runs as a long-polling bot with candle-aligned auto-signals.

Signals fire 15 seconds BEFORE each 5-min candle boundary so users
have time to act before the slot opens.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    MODEL_PATH, DATA_DIR, LOGS_DIR, PREDICTION_THRESHOLD,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TRADE_AMOUNT
)
from src.data_fetcher import fetch_live_candles
from src.features import compute_features, prepare_ml_data
from src.tracker import (
    record_signal, resolve_trade, get_stats,
    format_stats_message, format_recent_trades_message,
    load_tracker
)

# ─── Constants ─────────────────────────────────────────────────────────────────

SIGNAL_LEAD_SECONDS = 15   # fire signal this many seconds before slot opens
CANDLE_MINUTES = 5

# ─── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "telegram_bot.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Prediction helpers ─────────────────────────────────────────────────────────

def load_model():
    """Load trained XGBoost model and metrics."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return model, metrics


def get_live_prediction():
    """Get current direction prediction from model."""
    model, metrics = load_model()
    threshold = metrics.get("threshold", PREDICTION_THRESHOLD)

    df = fetch_live_candles(lookback=200)
    df_feat = compute_features(df.copy())
    X, _, _ = prepare_ml_data(df_feat, drop_na=True)

    X_last = X.iloc[[-1]]
    proba = model.predict_proba(X_last)[0, 1]
    prediction = 1 if proba >= threshold else 0
    confidence = abs(proba - 0.5) * 2

    last_c = df_feat.iloc[-1]
    prev_c = df_feat.iloc[-2]

    # The candle that just closed (we're predicting the NEXT one)
    closed_candle_up = 1 if last_c["close"] > last_c["open"] else 0

    result = {
        "prediction": "UP" if prediction == 1 else "DOWN",
        "direction_code": prediction,
        "probability_up": round(proba, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "last_close": round(last_c["close"], 2),
        "prev_close": round(prev_c["close"], 2),
        "closed_candle_up": closed_candle_up,
        "rsi": round(last_c["rsi"], 2) if not np.isnan(last_c["rsi"]) else None,
        "macd_histogram": round(last_c["histogram"], 4) if not np.isnan(last_c["histogram"]) else None,
        "volume_ratio": round(last_c["volume_ratio"], 2) if not np.isnan(last_c["volume_ratio"]) else None,
        "bb_position": round(last_c["bb_position"], 2) if not np.isnan(last_c["bb_position"]) else None,
        "atr_pct": round(last_c["atr_pct"], 4) if not np.isnan(last_c["atr_pct"]) else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return result, metrics


# ─── Candle-aligned slot computation ────────────────────────────────────────────

def _next_candle_boundary():
    """
    Return the next exact 5-min candle boundary as a timezone-aware
    UTC datetime.  E.g. if now is 12:03:22 -> returns 12:05:00.
    If now is exactly 12:05:00.000 -> returns 12:10:00.
    """
    now = datetime.now(timezone.utc)
    # Truncate to current minute
    base = now.replace(second=0, microsecond=0)
    remainder = base.minute % CANDLE_MINUTES
    if remainder == 0 and now.second == 0 and now.microsecond == 0:
        # Exactly on boundary — next boundary is 5 min later
        return base + timedelta(minutes=CANDLE_MINUTES)
    else:
        return base + timedelta(minutes=CANDLE_MINUTES - remainder)


def get_next_slot_times():
    """
    Return (slot_open_iso, slot_close_iso, slot_open_display, slot_close_display,
            mins_until_open, secs_until_open).

    slot_open/close _iso  : full ISO-8601 strings for storage (no ambiguity)
    slot_open/close _display : "HH:MM UTC" for human-readable messages
    """
    slot_open_dt = _next_candle_boundary()
    slot_close_dt = slot_open_dt + timedelta(minutes=CANDLE_MINUTES)

    now = datetime.now(timezone.utc)
    time_until = (slot_open_dt - now).total_seconds()
    mins = int(time_until // 60)
    secs = int(time_until % 60)

    return (
        slot_open_dt.strftime("%Y-%m-%dT%H:%M:%S"),   # ISO for storage
        slot_close_dt.strftime("%Y-%m-%dT%H:%M:%S"),   # ISO for storage
        slot_open_dt.strftime("%H:%M UTC"),              # display
        slot_close_dt.strftime("%H:%M UTC"),             # display
        mins,
        secs,
    )


def format_signal_message(result, metrics=None, stats=None):
    """Format prediction as a Telegram message."""
    direction_str = result["prediction"]
    direction_emoji = "\U0001f4c8" if result["direction_code"] == 1 else "\U0001f4c9"

    strength = "HIGH" if result["confidence"] >= 0.7 else \
               "MEDIUM" if result["confidence"] >= 0.4 else \
               "LOW"

    _, _, slot_open_disp, slot_close_disp, mins, secs = get_next_slot_times()
    win_rate = metrics.get("validation_win_rate", 0) if metrics else 0
    ev = metrics.get("expected_value_per_dollar", 0) if metrics else 0

    msg = (
        f"BTC 5m Signal -- {slot_open_disp} -> {slot_close_disp}\n"
        f"---\n"
        f"Direction: {direction_str} {direction_emoji}\n"
        f"Probability UP: {result['probability_up']:.1%}\n"
        f"Confidence: {result['confidence']:.1%}\n"
        f"---\n"
        f"Trade: ${TRADE_AMOUNT} -> {direction_str}\n"
        f"---\n"
        f"Indicators:\n"
        f"  RSI: {result['rsi']}\n"
        f"  MACD hist: {result['macd_histogram']}\n"
        f"  Vol ratio: {result['volume_ratio']}x\n"
        f"  BB pos: {result['bb_position']}\n"
        f"---\n"
        f"Expires in: {mins}m {secs}s\n"
        f"{strength} CONFIDENCE\n"
    )

    if win_rate:
        msg += f"Model win rate: {win_rate:.1%}\n"
        msg += f"EV per $1: ${ev:.4f}\n"

    # Live stats if available
    if stats and stats["total"] > 0:
        msg += (
            f"---\n"
            f"Live Stats:\n"
            f"  Trades: {stats['total']} | WR: {stats['win_rate']:.1%}\n"
            f"  P&L: ${stats['total_profit']:.2f}\n"
            f"  Streak: {stats['current_streak']} {stats['current_streak_type'].upper() if stats['current_streak_type'] else ''}"
        )

    return msg


# ─── Trade resolution (shared) ──────────────────────────────────────────────────

def resolve_pending_trades():
    """
    Resolve ALL expired unresolved trades by fetching candle data and
    matching on exact 5-min aligned timestamps.
    """
    now = datetime.now(timezone.utc)
    tracker_data = load_tracker()
    unresolved = [
        t for t in tracker_data.get("trades", [])
        if not t.get("resolved")
    ]

    if not unresolved:
        return

    # Filter to trades whose slot has closed
    expired = []
    for trade in unresolved:
        slot_close_dt = _parse_slot_time(trade["slot_close"])
        if now >= slot_close_dt:
            expired.append(trade)

    if not expired:
        return

    # Fetch candles once for all resolutions
    live_df = fetch_live_candles(lookback=100)

    for trade in expired:
        # Build the exact expected open timestamp for this trade's slot
        slot_open_dt = _parse_slot_time(trade["slot_open"])
        open_ts_ms = int(slot_open_dt.timestamp() * 1000)

        # Exact match: find the candle whose timestamp == slot open
        matched = live_df[live_df["timestamp"] == open_ts_ms]

        if matched.empty:
            # Fallback: allow up to 1 minute tolerance for exchange timestamp drift
            tolerance_ms = 60_000
            close_matches = live_df[
                (live_df["timestamp"] >= open_ts_ms - tolerance_ms) &
                (live_df["timestamp"] <= open_ts_ms + tolerance_ms)
            ]
            if close_matches.empty:
                log.warning(
                    f"Could not find candle for trade {trade['id']} "
                    f"(slot_open={trade['slot_open']}). Skipping."
                )
                continue
            matched = close_matches.iloc[[0]]
        else:
            matched = matched.iloc[[0]]

        candle = matched.iloc[0]
        outcome_code = 1 if candle["close"] > candle["open"] else 0

        resolved = resolve_trade(trade["id"], outcome_code)
        if resolved:
            log.info(f"Resolved trade {trade['id']}: {resolved['result']}")


def _parse_slot_time(slot_str):
    """
    Parse a slot time string to a timezone-aware UTC datetime.

    Supports two formats:
      1. ISO-8601:  "2026-03-19T19:30:00"   (new, unambiguous)
      2. Legacy:    "19:30 UTC"              (old trades — combined with today's date,
                                              with midnight-crossover guard)
    """
    # ── Try ISO-8601 first (new format, always unambiguous) ─────────────────
    if "T" in slot_str:
        dt = datetime.strptime(slot_str, "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)

    # ── Legacy "HH:MM UTC" format ──────────────────────────────────────────
    t = datetime.strptime(slot_str, "%H:%M UTC")
    now = datetime.now(timezone.utc)
    today = now.date()
    dt = datetime.combine(today, t.time(), tzinfo=timezone.utc)

    # Midnight-crossover guard:
    # If the parsed time is >12 hours in the future, it was probably yesterday.
    if (dt - now).total_seconds() > 12 * 3600:
        dt -= timedelta(days=1)
    # If the parsed time is >12 hours in the PAST, it's probably tomorrow.
    # (e.g. now is 23:58, slot_close is "00:00 UTC" -> should be tomorrow)
    elif (now - dt).total_seconds() > 12 * 3600:
        dt += timedelta(days=1)

    return dt


# ─── Telegram Bot ───────────────────────────────────────────────────────────────

def run_bot():
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")
        return

    # ── Auto-provision data + model if missing ─────────────────────────────
    from src.provision import provision, check_healthy
    if not check_healthy():
        log.info("Data or model missing — provisioning on startup...")
        provision(verbose=False)
    else:
        log.info("Data and model present — starting bot...")

    # ── Command handlers ─────────────────────────────────────────────────────

    async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "BTC 5-Min Signal Bot\n\n"
            "XGBoost ML-powered BTC direction predictions.\n\n"
            "/signal - Get a live prediction\n"
            "/stats - Live win/loss tracker\n"
            "/status - Model accuracy stats\n"
            "/accuracy - Detailed accuracy report\n"
            "/help - Help"
        )

    async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Help\n\n"
            "/signal - Live prediction\n"
            "/stats - Win/loss tracker\n"
            "/status - Model stats\n"
            "/accuracy - Detailed accuracy report\n\n"
            "Auto-signals every 5 min!"
        )

    async def signal_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text("Fetching latest data...")

            # Resolve ALL expired unresolved trades first
            resolve_pending_trades()

            # Get new prediction
            pred_result, metrics = get_live_prediction()
            stats = get_stats()
            slot_open_iso, slot_close_iso, _, _, _, _ = get_next_slot_times()

            trade = record_signal(
                slot_open_time=slot_open_iso,
                slot_close_time=slot_close_iso,
                direction=pred_result["prediction"],
                direction_code=pred_result["direction_code"],
                probability_up=pred_result["probability_up"],
                confidence=pred_result["confidence"],
                close_at_signal=pred_result["last_close"],
                rsi=pred_result["rsi"],
                macd_histogram=pred_result["macd_histogram"],
                volume_ratio=pred_result["volume_ratio"],
            )
            log.info(f"New signal recorded: id={trade['id']}, {trade['direction']}")

            msg = format_signal_message(pred_result, metrics, stats)
            await update.message.reply_text(msg)

            stats_msg = format_stats_message()
            await update.message.reply_text(stats_msg)

        except Exception as e:
            log.error(f"Signal error: {e}")
            import traceback
            log.error(traceback.format_exc())
            await update.message.reply_text(f"Error: {e}")

    async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            msg = format_stats_message()
            await update.message.reply_text(msg)

            recent = format_recent_trades_message(5)
            if recent:
                await update.message.reply_text(recent)
        except Exception as e:
            log.error(f"Stats error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            _, metrics = load_model()
            val_acc = metrics.get("validation_accuracy", 0)
            val_auc = metrics.get("validation_auc", 0)
            win_rate = metrics.get("validation_win_rate", 0)
            ev = metrics.get("expected_value_per_dollar", 0)
            total_trades = metrics.get("total_validation_trades", 0)
            training_samples = metrics.get("training_samples", 0)

            model_date = datetime.fromtimestamp(
                os.path.getmtime(MODEL_PATH)
            ).strftime("%Y-%m-%d %H:%M") if os.path.exists(MODEL_PATH) else "Unknown"

            status = (
                "Model Status\n"
                "---\n"
                f"Model trained: {model_date}\n"
                f"Training samples: {training_samples:,}\n"
                f"Validation trades: {total_trades:,}\n"
                "---\n"
                f"Validation Accuracy: {val_acc:.2%}\n"
                f"Validation AUC: {val_auc:.4f}\n"
                f"Win Rate (UP): {win_rate:.2%}\n"
                f"Threshold: {metrics.get('threshold', PREDICTION_THRESHOLD):.3f}\n"
                f"Expected Value: ${ev:.4f}/trade\n"
                "---\n"
            )
            status += "Model is profitable!" if val_acc >= 0.52 else "Below breakeven"

            await update.message.reply_text(status)
        except Exception as e:
            log.error(f"Status error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def accuracy_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Detailed accuracy report based on live trade history."""
        try:
            # Resolve any pending trades first
            resolve_pending_trades()

            tracker_data = load_tracker()
            trades = tracker_data.get("trades", [])
            resolved_trades = [t for t in trades if t.get("resolved")]

            if not resolved_trades:
                await update.message.reply_text("No resolved trades yet. Place some trades first!")
                return

            total = len(resolved_trades)
            wins = sum(1 for t in resolved_trades if t["result"] == "WIN")
            losses = total - wins
            win_rate = wins / total if total > 0 else 0

            # Breakdown by confidence level
            high_conf = [t for t in resolved_trades if t.get("confidence", 0) >= 0.7]
            med_conf = [t for t in resolved_trades if 0.4 <= t.get("confidence", 0) < 0.7]
            low_conf = [t for t in resolved_trades if t.get("confidence", 0) < 0.4]

            def calc_wr(trades_list):
                if not trades_list:
                    return 0, 0
                w = sum(1 for t in trades_list if t["result"] == "WIN")
                return w / len(trades_list), len(trades_list)

            high_wr, high_n = calc_wr(high_conf)
            med_wr, med_n = calc_wr(med_conf)
            low_wr, low_n = calc_wr(low_conf)

            # Breakdown by direction
            up_trades = [t for t in resolved_trades if t["direction_code"] == 1]
            down_trades = [t for t in resolved_trades if t["direction_code"] == 0]
            up_wr, up_n = calc_wr(up_trades)
            down_wr, down_n = calc_wr(down_trades)

            # P&L
            total_profit = sum(t.get("profit", 0) for t in resolved_trades)
            ev_per_trade = total_profit / total if total > 0 else 0

            # Model metrics
            _, metrics = load_model()

            msg = (
                "Detailed Accuracy Report\n"
                "===\n"
                f"Total resolved trades: {total}\n"
                f"Wins: {wins} | Losses: {losses}\n"
                f"Live Win Rate: {win_rate:.1%}\n"
                f"Total P&L: ${total_profit:.2f}\n"
                f"EV per trade: ${ev_per_trade:.4f}\n"
                "---\n"
                "By Confidence:\n"
                f"  HIGH (>=70%):   {high_wr:.1%} ({high_n} trades)\n"
                f"  MEDIUM (40-70%): {med_wr:.1%} ({med_n} trades)\n"
                f"  LOW (<40%):     {low_wr:.1%} ({low_n} trades)\n"
                "---\n"
                "By Direction:\n"
                f"  UP signals:   {up_wr:.1%} ({up_n} trades)\n"
                f"  DOWN signals: {down_wr:.1%} ({down_n} trades)\n"
                "---\n"
                "Model Validation:\n"
                f"  Accuracy: {metrics.get('validation_accuracy', 0):.2%}\n"
                f"  AUC: {metrics.get('validation_auc', 0):.4f}\n"
                f"  Threshold: {metrics.get('threshold', PREDICTION_THRESHOLD):.3f}\n"
            )

            breakeven = 1 / (1 + 0.96)
            if win_rate >= breakeven:
                msg += f"\nProfitable! (breakeven = {breakeven:.1%})"
            else:
                msg += f"\nBelow breakeven ({breakeven:.1%})"

            await update.message.reply_text(msg)

        except Exception as e:
            log.error(f"Accuracy error: {e}")
            import traceback
            log.error(traceback.format_exc())
            await update.message.reply_text(f"Error: {e}")

    async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Try /start, /signal, /stats, /status, or /accuracy.")

    # ── Candle-aligned auto-signal job ───────────────────────────────────────
    #
    # Instead of run_repeating (which drifts from candle boundaries), we use
    # run_once to schedule each signal exactly SIGNAL_LEAD_SECONDS before the
    # next 5-min boundary.  After sending, the callback re-schedules itself
    # for the following boundary.
    #
    # Timeline example (SIGNAL_LEAD_SECONDS = 15):
    #   12:04:45  ->  signal fires (predicting 12:05-12:10 slot)
    #   12:09:45  ->  signal fires (predicting 12:10-12:15 slot)
    #   12:14:45  ->  signal fires (predicting 12:15-12:20 slot)
    #
    # The prediction uses the last *closed* candle from MEXC.  At 12:04:45
    # the latest closed candle is 11:55-12:00 (the 12:00-12:05 candle is
    # still open).  This is correct — the model was trained to predict the
    # NEXT candle from the features of the last closed candle.

    async def send_auto_signal(ctx: ContextTypes.DEFAULT_TYPE):
        """Candle-aligned signal callback. Sends signal then re-schedules."""
        try:
            # 1. Resolve ALL unresolved trades that have expired
            resolve_pending_trades()

            # 2. Get prediction (uses latest complete candle)
            pred_result, metrics = get_live_prediction()
            stats = get_stats()
            slot_open_iso, slot_close_iso, _, _, _, _ = get_next_slot_times()

            # 3. Record new signal for the upcoming slot
            trade = record_signal(
                slot_open_time=slot_open_iso,
                slot_close_time=slot_close_iso,
                direction=pred_result["prediction"],
                direction_code=pred_result["direction_code"],
                probability_up=pred_result["probability_up"],
                confidence=pred_result["confidence"],
                close_at_signal=pred_result["last_close"],
                rsi=pred_result["rsi"],
                macd_histogram=pred_result["macd_histogram"],
                volume_ratio=pred_result["volume_ratio"],
            )

            # 4. Send signal message
            msg = format_signal_message(pred_result, metrics, stats)
            await ctx.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
            )
            log.info(
                f"Auto-signal sent: id={trade['id']}, {pred_result['prediction']}, "
                f"prob={pred_result['probability_up']:.4f}, "
                f"slot={slot_open_iso}"
            )

        except Exception as e:
            log.error(f"Auto-signal error: {e}")
            import traceback
            log.error(traceback.format_exc())

        finally:
            # Always re-schedule the next signal, even if this one errored
            _schedule_next_signal(ctx.job.data["app"])

    def _schedule_next_signal(app):
        """
        Schedule `send_auto_signal` to fire SIGNAL_LEAD_SECONDS before the
        next 5-min candle boundary.
        """
        now = datetime.now(timezone.utc)
        next_boundary = _next_candle_boundary()
        fire_at = next_boundary - timedelta(seconds=SIGNAL_LEAD_SECONDS)

        # If fire_at is already in the past (e.g. bot just started at XX:X4:50),
        # skip to the boundary after that
        if fire_at <= now:
            next_boundary = next_boundary + timedelta(minutes=CANDLE_MINUTES)
            fire_at = next_boundary - timedelta(seconds=SIGNAL_LEAD_SECONDS)

        delay = (fire_at - now).total_seconds()
        log.info(
            f"Next signal scheduled for {fire_at.strftime('%H:%M:%S UTC')} "
            f"(in {delay:.0f}s, slot {next_boundary.strftime('%H:%M UTC')})"
        )

        app.job_queue.run_once(
            send_auto_signal,
            when=delay,
            data={"app": app},
        )

    # ── Build & start ────────────────────────────────────────────────────────

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("accuracy", accuracy_cmd))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))

    # Schedule the first candle-aligned signal
    _schedule_next_signal(app)

    log.info("Bot started with candle-aligned auto-signals.")
    app.run_polling()


if __name__ == "__main__":
    print("Starting Telegram bot...")
    run_bot()
