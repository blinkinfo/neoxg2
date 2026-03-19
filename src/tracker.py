"""
tracker.py
Tracks signal history, calculates wins/losses, win rate, streaks.
Stores results in a JSON file atomically.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from src.config import DATA_DIR, LOGS_DIR, PAYOUT

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "tracker.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

TRACKER_FILE = os.path.join(DATA_DIR, "signal_results.json")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _display_slot(slot_str: str) -> str:
    """
    Convert slot time string to display format.
    ISO-8601 '2026-03-19T19:30:00' -> '19:30 UTC'
    Legacy   '19:30 UTC'           -> '19:30 UTC' (unchanged)
    """
    if "T" in slot_str:
        try:
            dt = datetime.strptime(slot_str, "%Y-%m-%dT%H:%M:%S")
            return dt.strftime("%H:%M UTC")
        except ValueError:
            return slot_str
    return slot_str


def _pnl_display(amount: float) -> str:
    """Format P&L with sign prefix."""
    if amount >= 0:
        return f"+${amount:.2f}"
    return f"-${abs(amount):.2f}"


def _bar(value: float, width: int = 12) -> str:
    """Unicode block bar for visual meters (0.0 to 1.0)."""
    filled = round(value * width)
    empty = width - filled
    return "\u2593" * filled + "\u2591" * empty


def _streak_icon(streak_type: str) -> str:
    if streak_type and streak_type.lower() == "win":
        return "\u25b2"  # triangle up
    elif streak_type and streak_type.lower() == "loss":
        return "\u25bc"  # triangle down
    return "\u2014"  # em dash


# ── Persistence ───────────────────────────────────────────────────────────────

def load_tracker() -> dict:
    """Load tracker JSON. Backs up and resets on corruption."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            ts = int(datetime.now(timezone.utc).timestamp())
            backup = TRACKER_FILE + f".broken_{ts}"
            os.rename(TRACKER_FILE, backup)
            log.warning(f"Corrupted tracker file backed up to {backup}")
    return {
        "trades": [],
        "stats": {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "current_streak": 0,
            "current_streak_type": None,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "total_profit": 0.0,
            "payout": PAYOUT,
        },
    }


def save_tracker(data: dict) -> None:
    """Save tracker JSON atomically (write temp, then rename)."""
    tmp = TRACKER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, TRACKER_FILE)


# ── Write ops ─────────────────────────────────────────────────────────────────

def record_signal(
    slot_open_time: str,
    slot_close_time: str,
    direction: str,
    direction_code: int,
    probability_up: float,
    confidence: float,
    close_at_signal: float,
    rsi: Optional[float] = None,
    macd_histogram: Optional[float] = None,
    volume_ratio: Optional[float] = None,
) -> dict:
    """Record a new signal. Result filled in later when candle closes."""
    data = load_tracker()
    trade = {
        "id":               len(data["trades"]) + 1,
        "slot_open":        slot_open_time,
        "slot_close":       slot_close_time,
        "direction":        direction,
        "direction_code":   int(direction_code),
        "probability_up":   float(probability_up),
        "confidence":       float(confidence),
        "close_at_signal":  float(close_at_signal),
        "rsi":              float(rsi) if rsi is not None else None,
        "macd_histogram":   float(macd_histogram) if macd_histogram is not None else None,
        "volume_ratio":     float(volume_ratio) if volume_ratio is not None else None,
        "result":           None,
        "profit":           0.0,
        "resolved":         False,
        "recorded_at":      datetime.now(timezone.utc).isoformat(),
        "resolved_at":      None,
    }
    data["trades"].append(trade)
    save_tracker(data)
    return trade


def resolve_trade(trade_id: int, outcome_code: int) -> Optional[dict]:
    """
    Resolve a trade after candle close.
    outcome_code: 1 = candle UP, 0 = candle DOWN
    """
    data = load_tracker()
    for trade in reversed(data["trades"]):
        if trade["id"] != trade_id:
            continue
        if trade["resolved"]:
            return None

        won = trade["direction_code"] == outcome_code
        trade["result"]      = "WIN" if won else "LOSS"
        trade["profit"]      = PAYOUT if won else -1.00
        trade["resolved"]    = True
        trade["resolved_at"] = datetime.now(timezone.utc).isoformat()

        stats = data["stats"]
        stats["total"] += 1
        if won:
            stats["wins"] += 1
            if stats["current_streak_type"] == "win":
                stats["current_streak"] += 1
            else:
                stats["current_streak"]      = 1
                stats["current_streak_type"] = "win"
            stats["max_win_streak"] = max(stats["max_win_streak"], stats["current_streak"])
        else:
            stats["losses"] += 1
            if stats["current_streak_type"] == "loss":
                stats["current_streak"] += 1
            else:
                stats["current_streak"]      = 1
                stats["current_streak_type"] = "loss"
            stats["max_loss_streak"] = max(stats["max_loss_streak"], stats["current_streak"])

        stats["win_rate"]     = stats["wins"] / stats["total"]
        stats["total_profit"] = round(stats["total_profit"] + trade["profit"], 4)

        save_tracker(data)
        return trade

    return None


# ── Read ops ──────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    return load_tracker()["stats"]


def get_recent_trades(n: int = 10) -> list:
    return load_tracker()["trades"][-n:]


# ── Formatting (HTML) ─────────────────────────────────────────────────────────

def format_stats_message() -> str:
    """Session stats card for /stats — HTML formatted."""
    stats = get_stats()
    total = stats["total"]

    if total == 0:
        return (
            "<b>\u2593 Session Tracker</b>\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            "No resolved trades yet.\n\n"
            "<i>Auto-signals fire every 5 min.\n"
            "Trades resolve when their candle closes.\n"
            "Check back in a few minutes!</i>"
        )

    wr     = stats["win_rate"]
    pnl    = stats["total_profit"]
    ev_pt  = pnl / total
    stype  = stats["current_streak_type"] or ""
    sk     = stats["current_streak"]
    sk_icon = _streak_icon(stype)

    # Win rate bar
    wr_bar = _bar(wr)

    breakeven = 1 / (1 + PAYOUT)
    if wr >= breakeven:
        verdict = "\u25cf PROFITABLE"
    else:
        verdict = "\u25cf BELOW BREAKEVEN"

    lines = [
        "<b>\u2593 Session Tracker</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"  <b>{_pnl_display(pnl)}</b>  Total P&amp;L",
        f"  {wr_bar}  <b>{wr:.1%}</b>  Win Rate",
        "",
        "<b>Record</b>",
        "<code>"
        f"  Trades          {total}\n"
        f"  Wins            {stats['wins']}\n"
        f"  Losses          {stats['losses']}\n"
        f"  EV per trade    {ev_pt:+.4f}"
        "</code>",
        "",
        "<b>Streaks</b>",
        "<code>"
        f"  Current         {sk_icon} {sk} {stype.upper() if stype else '--'}\n"
        f"  Best win        {stats['max_win_streak']}\n"
        f"  Worst loss      {stats['max_loss_streak']}"
        "</code>",
        "",
        f"<b>Payout</b>  +${PAYOUT:.2f} (win)  /  -$1.00 (loss)",
        "",
        f"<b>{verdict}</b>",
    ]
    return "\n".join(lines)


def format_recent_trades_message(n: int = 5) -> str:
    """Last N trades as a formatted list — HTML."""
    trades = get_recent_trades(n)
    if not trades:
        return ""

    lines = [
        "<b>\u23f1 Recent Trades</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
    ]

    for t in reversed(trades):
        slot  = _display_slot(t["slot_open"])
        prob  = t["probability_up"]
        direc = t["direction"]
        tid   = t["id"]

        if t["resolved"]:
            if t["result"] == "WIN":
                icon   = "\u2713"  # checkmark
                profit = t["profit"]
                result_str = f"<b>{_pnl_display(profit)}</b>"
            else:
                icon   = "\u2717"  # x mark
                profit = t["profit"]
                result_str = f"{_pnl_display(profit)}"

            lines.append(
                f"  {icon}  <code>#{tid:>3}</code>  {slot}  "
                f"<b>{direc:<4}</b>  {prob:.0%}  {result_str}"
            )
        else:
            lines.append(
                f"  \u25cb  <code>#{tid:>3}</code>  {slot}  "
                f"<b>{direc:<4}</b>  {prob:.0%}  <i>pending</i>"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    print(format_stats_message())
