"""
tracker.py
Tracks signal history, calculates wins/losses, win rate, streaks.
Stores results in a JSON file.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.config import DATA_DIR, LOGS_DIR

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "tracker.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


TRACKER_FILE = os.path.join(DATA_DIR, "signal_results.json")


def load_tracker():
    """Load tracker data from JSON file. Handles corrupted files gracefully."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # File corrupted — back it up and start fresh
            backup = TRACKER_FILE + f".broken_{datetime.now().strftime('%s')}"
            os.rename(TRACKER_FILE, backup)
            log.warning(f"Corrupted tracker file renamed to {backup}")
    return {
        "trades": [],
        "stats": {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "current_streak": 0, "current_streak_type": None,
            "max_win_streak": 0, "max_loss_streak": 0,
            "total_profit": 0.0, "payout": 0.96,
        }
    }


def save_tracker(data):
    """Save tracker data to JSON file atomically (write to temp, then rename)."""
    tmp_path = TRACKER_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, TRACKER_FILE)  # atomic on POSIX


def record_signal(
    slot_open_time: str,
    slot_close_time: str,
    direction: str,          # "UP" or "DOWN"
    direction_code: int,     # 1 = UP, 0 = DOWN
    probability_up: float,
    confidence: float,
    close_at_signal: float,  # close price when signal was generated
    rsi: Optional[float] = None,
    macd_histogram: Optional[float] = None,
    volume_ratio: Optional[float] = None,
):
    """
    Record a signal that was sent. Outcome will be filled in later
    when the candle closes and we fetch the actual result.
    """
    data = load_tracker()
    
    trade = {
        "id": len(data["trades"]) + 1,
        "slot_open": slot_open_time,
        "slot_close": slot_close_time,
        "direction": direction,
        "direction_code": int(direction_code),
        "probability_up": float(probability_up),
        "confidence": float(confidence),
        "close_at_signal": float(close_at_signal),
        "rsi": float(rsi) if rsi is not None else None,
        "macd_histogram": float(macd_histogram) if macd_histogram is not None else None,
        "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
        "result": None,        # will be "WIN" or "LOSS"
        "profit": 0.0,         # +0.96 for win, -1.00 for loss
        "resolved": False,
        "recorded_at": datetime.utcnow().isoformat(),
        "resolved_at": None,
    }
    
    data["trades"].append(trade)
    save_tracker(data)
    
    return trade


def resolve_trade(trade_id: int, outcome_code: int):
    """
    Resolve a trade after the candle closes.
    
    outcome_code: 1 = candle went UP, 0 = candle went DOWN
    """
    data = load_tracker()
    
    for trade in reversed(data["trades"]):
        if trade["id"] == trade_id:
            if trade["resolved"]:
                return  # already resolved
            
            won = trade["direction_code"] == outcome_code
            trade["result"] = "WIN" if won else "LOSS"
            trade["profit"] = 0.96 if won else -1.00
            trade["resolved"] = True
            trade["resolved_at"] = datetime.utcnow().isoformat()
            
            # Update stats
            stats = data["stats"]
            stats["total"] += 1
            
            if won:
                stats["wins"] += 1
                # Extend streak
                if stats["current_streak_type"] == "win":
                    stats["current_streak"] += 1
                else:
                    stats["current_streak"] = 1
                    stats["current_streak_type"] = "win"
                stats["max_win_streak"] = max(stats["max_win_streak"], stats["current_streak"])
            else:
                stats["losses"] += 1
                # Extend losing streak
                if stats["current_streak_type"] == "loss":
                    stats["current_streak"] += 1
                else:
                    stats["current_streak"] = 1
                    stats["current_streak_type"] = "loss"
                stats["max_loss_streak"] = max(stats["max_loss_streak"], stats["current_streak"])
            
            stats["win_rate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            stats["total_profit"] += trade["profit"]
            
            save_tracker(data)
            return trade
    
    return None


def get_stats():
    """Return current stats summary."""
    data = load_tracker()
    return data["stats"]


def get_recent_trades(n=10):
    """Return the most recent n trades."""
    data = load_tracker()
    return data["trades"][-n:]


def format_stats_message():
    """Format stats as a Telegram message."""
    stats = get_stats()
    
    streak_emoji = "🟢" if stats["current_streak_type"] == "win" else "🔴"
    streak_text = (
        f"{streak_emoji} {stats['current_streak']} {stats['current_streak_type'].upper()}"
        if stats["current_streak_type"]
        else "—"
    )
    
    total = stats["total"]
    if total == 0:
        return "📊 *No trades recorded yet.*\nPlace some trades first!"
    
    ev_per_trade = stats["total_profit"] / total
    
    msg = (
        "📊 *Signal Tracker*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Total trades:  {total}\n"
        f"Wins:         {stats['wins']}  |  Losses: {stats['losses']}\n"
        f"Win rate:     {stats['win_rate']:.1%}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Current streak: {streak_text}\n"
        f"Max win streak: {stats['max_win_streak']}\n"
        f"Max loss streak: {stats['max_loss_streak']}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Total P&L:    ${stats['total_profit']:.2f}\n"
        f"EV per trade: ${ev_per_trade:.4f}\n"
        f"Payout:       $0.96 (win) / -$1.00 (loss)"
    )
    
    return msg


def format_recent_trades_message(n=5):
    """Format recent trades as a Telegram message."""
    trades = get_recent_trades(n)
    if not trades:
        return ""
    
    lines = ["\n━━━━━━━━━━━━━━━\n📋 *Recent Trades:*\n"]
    
    for t in reversed(trades):
        result_emoji = "✅" if t["result"] == "WIN" else "❌"
        resolved = t["resolved"]
        
        if resolved:
            lines.append(
                f"{result_emoji} {t['slot_open']} | "
                f"{t['direction']} | "
                f"P={t['probability_up']:.1%} | "
                f"{'WIN' if t['result']=='WIN' else 'LOSS'} ${t['profit']:.2f}"
            )
        else:
            lines.append(
                f"⏳ {t['slot_open']} | "
                f"{t['direction']} | "
                f"P={t['probability_up']:.1%} | "
                f"PENDING"
            )
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print(format_stats_message())
