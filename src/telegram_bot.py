"""
telegram_bot.py
Signal bot — sends BTC 5-min direction predictions to Telegram.
Runs as a long-polling bot with periodic auto-signals.
"""

import os
import sys
import json
import logging
from datetime import datetime
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
    
    metrics_path = MODEL_PATH.replace(".json", "_metrics.json")
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
        "prediction": "UP 📈" if prediction == 1 else "DOWN 📉",
        "direction_code": prediction,
        "probability_up": round(proba, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "last_close": round(last_c["close"], 2),
        "prev_close": round(prev_c["close"], 2),
        "closed_candle_up": closed_candle_up,  # result of last complete candle
        "rsi": round(last_c["rsi"], 2) if not np.isnan(last_c["rsi"]) else None,
        "macd_histogram": round(last_c["histogram"], 4) if not np.isnan(last_c["histogram"]) else None,
        "volume_ratio": round(last_c["volume_ratio"], 2) if not np.isnan(last_c["volume_ratio"]) else None,
        "bb_position": round(last_c["bb_position"], 2) if not np.isnan(last_c["bb_position"]) else None,
        "atr_pct": round(last_c["atr_pct"], 4) if not np.isnan(last_c["atr_pct"]) else None,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    return result, metrics


def get_next_slot_times():
    """Return (slot_open_str, slot_close_str, mins_until_open, secs_until_open)."""
    now = datetime.utcnow()
    next_open = now.replace(second=0, microsecond=0)
    
    # Round up to next 5-min mark
    minute = next_open.minute
    second = now.second
    remainder = (minute % 5, second)
    if remainder == (0, 0) and now.microsecond == 0:
        add_minutes = 0  # exactly on a 5-min boundary
    else:
        add_minutes = 5 - (minute % 5)
    
    next_open = next_open + pd.Timedelta(minutes=add_minutes)
    next_close = next_open + pd.Timedelta(minutes=5)
    
    slot_open_str = next_open.strftime("%H:%M UTC")
    slot_close_str = next_close.strftime("%H:%M UTC")
    
    time_until = (next_open - now).total_seconds()
    mins = int(time_until // 60)
    secs = int(time_until % 60)
    
    return slot_open_str, slot_close_str, mins, secs


def format_signal_message(result, metrics=None, stats=None):
    """Format prediction as a Telegram message."""
    direction = result["prediction"]
    direction_str = "UP" if result["direction_code"] == 1 else "DOWN"
    emoji = "🟢" if result["direction_code"] == 1 else "🔴"
    
    strength = "🔥 HIGH" if result["confidence"] >= 0.7 else \
               "⚡ MEDIUM" if result["confidence"] >= 0.4 else \
               "🌫️ LOW"
    
    slot_open, slot_close, mins, secs = get_next_slot_times()
    win_rate = metrics.get("validation_win_rate", 0) if metrics else 0
    ev = metrics.get("expected_value_per_dollar", 0) if metrics else 0
    
    msg = (
        f"⏰ BTC 5m Signal — {slot_open} → {slot_close}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Direction: {direction}\n"
        f"Probability UP: {result['probability_up']:.1%}\n"
        f"Confidence: {result['confidence']:.1%}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💰 Trade: ${TRADE_AMOUNT} → {direction_str}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 Indicators:\n"
        f"  RSI: {result['rsi']}\n"
        f"  MACD hist: {result['macd_histogram']}\n"
        f"  Vol ratio: {result['volume_ratio']}x\n"
        f"  BB pos: {result['bb_position']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⏱️  Expires in: {mins}m {secs}s\n"
        f"{strength} CONFIDENCE\n"
    )
    
    if win_rate:
        msg += f"📈 Model win rate: {win_rate:.1%}\n"
        msg += f"💵 EV per $1: ${ev:.4f}\n"
    
    # Live stats if available
    if stats and stats["total"] > 0:
        msg += (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Live Stats:\n"
            f"  Trades: {stats['total']} | WR: {stats['win_rate']:.1%}\n"
            f"  P&L: ${stats['total_profit']:.2f}\n"
            f"  Streak: {stats['current_streak']} {stats['current_streak_type'].upper() if stats['current_streak_type'] else ''}"
        )
    
    return msg


# ─── Telegram Bot ───────────────────────────────────────────────────────────────

def run_bot():
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes, JobQueue
    )
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")
        return

    # ── Auto-provision data + model if missing ─────────────────────────────
    from src.provision import provision, check_healthy
    if not check_healthy():
        log.info("Data or model missing — provisioning on startup...")
        provision(verbose=False)  # silent, logs already go to file
    else:
        log.info("Data and model present — starting bot...")

    # ── Build & start ────────────────────────────────────────────────────────
    
    async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 *BTC 5-Min Signal Bot*\n\n"
            "XGBoost ML-powered BTC direction predictions.\n\n"
            "/signal - Get a live prediction\n"
            "/stats - Live win/loss tracker\n"
            "/status - Model accuracy stats\n"
            "/help - Help",
            parse_mode="Markdown"
        )
    
    async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 *Help*\n\n"
            "/signal - Live prediction\n"
            "/stats - Win/loss tracker\n"
            "/status - Model stats\n\n"
            "Auto-signals every 5 min! 🕐"
        )
    
    async def signal_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text("🔄 Fetching latest data...")
            
            now = datetime.utcnow()
            
            # Resolve ALL expired unresolved trades first
            def parse_slot_close(slot_close_str):
                t = datetime.strptime(slot_close_str, "%H:%M UTC")
                today = datetime.utcnow().date()
                return datetime.combine(today, t.time())
            
            tracker_data = load_tracker()
            for trade in tracker_data.get("trades", []):
                if trade.get("resolved"):
                    continue
                slot_close_dt = parse_slot_close(trade["slot_close"])
                if now < slot_close_dt:
                    continue
                
                live_df = fetch_live_candles(lookback=50)
                live_df_feat = compute_features(live_df.copy())
                open_time = datetime.strptime(trade["slot_open"], "%H:%M UTC")
                open_today = datetime.combine(datetime.utcnow().date(), open_time.time())
                open_ts = int(open_today.timestamp() * 1000)
                live_df_feat["ts_diff"] = abs(live_df_feat["timestamp"] - open_ts)
                matched = live_df_feat.loc[live_df_feat["ts_diff"].idxmin()]
                outcome_code = 1 if matched["close"] > matched["open"] else 0
                resolved = resolve_trade(trade["id"], outcome_code)
                if resolved:
                    log.info(f"Resolved trade {trade['id']}: {resolved['result']}")
            
            # Get new prediction
            pred_result, metrics = get_live_prediction()
            stats = get_stats()
            slot_open, slot_close, _, _ = get_next_slot_times()
            
            trade = record_signal(
                slot_open_time=slot_open,
                slot_close_time=slot_close,
                direction="UP" if pred_result["direction_code"] == 1 else "DOWN",
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
            await update.message.reply_text(msg, parse_mode="Markdown")
            
            stats_msg = format_stats_message()
            await update.message.reply_text(stats_msg, parse_mode="Markdown")
            
        except Exception as e:
            log.error(f"Signal error: {e}")
            import traceback
            log.error(traceback.format_exc())
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            msg = format_stats_message()
            await update.message.reply_text(msg, parse_mode="Markdown")
            
            recent = format_recent_trades_message(5)
            if recent:
                await update.message.reply_text(recent, parse_mode="Markdown")
        except Exception as e:
            log.error(f"Stats error: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
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
                "📊 *Model Status*\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"Model trained: {model_date}\n"
                f"Training samples: {training_samples:,}\n"
                f"Validation trades: {total_trades:,}\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"Validation Accuracy: {val_acc:.2%}\n"
                f"Validation AUC: {val_auc:.4f}\n"
                f"Win Rate (UP): {win_rate:.2%}\n"
                f"Threshold: {metrics.get('threshold', PREDICTION_THRESHOLD):.3f}\n"
                f"Expected Value: ${ev:.4f}/trade\n"
                "━━━━━━━━━━━━━━━━━━\n"
            )
            status += "✅ Model is profitable!" if val_acc >= 0.52 else "⚠️ Below breakeven"
            
            await update.message.reply_text(status, parse_mode="Markdown")
        except Exception as e:
            log.error(f"Status error: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("❓ Try /start, /signal, /stats, or /status.")
    
    # ── Auto-signal job ──────────────────────────────────────────────────────
    
    def parse_slot_close(slot_close_str):
        """Parse 'HH:MM UTC' string to today's datetime (UTC)."""
        t = datetime.strptime(slot_close_str, "%H:%M UTC")
        today = datetime.utcnow().date()
        return datetime.combine(today, t.time(), tzinfo=None)
    
    async def send_auto_signal(app):
        try:
            now = datetime.utcnow()
            
            # 1. Get prediction FIRST (uses latest complete candle)
            pred_result, metrics = get_live_prediction()
            stats = get_stats()
            slot_open, slot_close_str, _, _ = get_next_slot_times()
            
            # 2. Resolve ALL unresolved trades that have expired
            tracker_data = load_tracker()
            for trade in tracker_data.get("trades", []):
                if trade.get("resolved"):
                    continue
                
                slot_close_dt = parse_slot_close(trade["slot_close"])
                
                if now < slot_close_dt:
                    continue  # candle hasn't closed yet, skip
                
                # Candle has closed — fetch it to determine outcome
                # We predicted for slot_open → slot_close
                # At slot_close, the candle from slot_open has closed
                # e.g. slot_open="11:45", slot_close="11:50"
                # We need to check: did the 11:45-11:50 candle go UP or DOWN?
                live_df = fetch_live_candles(lookback=50)
                live_df_feat = compute_features(live_df.copy())
                
                # Find the candle that matches slot_open time
                # slot_open like "11:45 UTC" — parse the time
                open_time = datetime.strptime(trade["slot_open"], "%H:%M UTC")
                open_today = datetime.combine(datetime.utcnow().date(), open_time.time())
                open_ts = int(open_today.timestamp() * 1000)
                
                # Find this candle in the dataframe (closest timestamp)
                live_df_feat["ts_diff"] = abs(live_df_feat["timestamp"] - open_ts)
                matched = live_df_feat.loc[live_df_feat["ts_diff"].idxmin()]
                
                outcome_code = 1 if matched["close"] > matched["open"] else 0
                
                resolved = resolve_trade(trade["id"], outcome_code)
                if resolved:
                    log.info(f"Resolved trade {trade['id']}: {resolved['result']}")
            
            # 3. Record new signal for the next window
            trade = record_signal(
                slot_open_time=slot_open,
                slot_close_time=slot_close_str,
                direction="UP" if pred_result["direction_code"] == 1 else "DOWN",
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
            await app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode="Markdown"
            )
            log.info(f"Auto-signal sent: id={trade['id']}, {pred_result['prediction']}, prob={pred_result['probability_up']:.4f}")
            
        except Exception as e:
            log.error(f"Auto-signal error: {e}")
            import traceback
            log.error(traceback.format_exc())
    
    # ── Build & start ────────────────────────────────────────────────────────
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    
    app.job_queue.run_repeating(
        send_auto_signal,
        interval=300,  # 5 minutes
        first=10
    )
    
    log.info("Bot started with tracker enabled.")
    app.run_polling()


if __name__ == "__main__":
    print("Starting Telegram bot...")
    run_bot()
