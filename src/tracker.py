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


def _pnl_emoji(amount: float) -> str:
    """P&L with emoji prefix."""
    if amount > 0:
        return f"\U0001f4b0 +${amount:.2f}"
    elif amount < 0:
        return f"\U0001f534 -${abs(amount):.2f}"
    return "\u26aa $0.00"


def _bar(value: float, width: int = 10) -> str:
    """Green/gray square bar for visual meters (0.0 to 1.0)."""
    filled = round(value * width)
    empty = width - filled
    return "\U0001f7e9" * filled + "\u2b1c" * empty


def _streak_display(streak: int, streak_type: str) -> str:
    """Streak with emoji."""
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
    """Session stats card for /stats -- emoji-rich HTML."""
    stats = get_stats()
    total = stats["total"]

    if total == 0:
        return (
            "\U0001f4ca  <b>Session Tracker</b>\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            "\U0001f4ed  No resolved trades yet.\n\n"
            "\U0001f552  <i>Auto-signals fire every 5 min.\n"
            "Trades resolve when their candle closes.\n"
            "Check back in a few minutes!</i>"
        )

    wr     = stats["win_rate"]
    pnl    = stats["total_profit"]
    ev_pt  = pnl / total
    stype  = stats["current_streak_type"] or ""
    sk     = stats["current_streak"]

    wr_bar  = _bar(wr)
    verdict = _verdict_line(wr)

    lines = [
        "\U0001f4ca  <b>Session Tracker</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
        f"      {_pnl_emoji(pnl)}  Total P&amp;L",
        f"      {wr_bar}  <b>{wr:.1%}</b>  Win Rate",
        f"      {verdict}",
        "",
        "\U0001f4c8  <b>Record</b>",
        "<code>"
        f"  Trades          {total}\n"
        f"  Wins            {stats['wins']}  \u2713\n"
        f"  Losses          {stats['losses']}  \u2717\n"
        f"  EV per trade    {ev_pt:+.4f}"
        "</code>",
        "",
        "\U0001f525  <b>Streaks</b>",
        "<code>"
        f"  Current         {_streak_display(sk, stype)}\n"
        f"  Best win        {stats['max_win_streak']}\n"
        f"  Worst loss      {stats['max_loss_streak']}"
        "</code>",
        "",
        f"\U0001f4b5  <b>Payout:</b>  +${PAYOUT:.2f} (win)  /  -$1.00 (loss)",
    ]
    return "\n".join(lines)


def format_recent_trades_message(n: int = 5) -> str:
    """Last N trades as a formatted list -- emoji-rich HTML."""
    trades = get_recent_trades(n)
    if not trades:
        return ""

    lines = [
        "\U0001f4dc  <b>Recent Trades</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
    ]

    for t in reversed(trades):
        slot  = _display_slot(t["slot_open"])
        prob  = t["probability_up"]
        direc = t["direction"]
        tid   = t["id"]

        if direc == "UP":
            dir_icon = "\U0001f53c"
        else:
            dir_icon = "\U0001f53d"

        if t["resolved"]:
            if t["result"] == "WIN":
                result_icon = "\u2705"
                profit = t["profit"]
                result_str = f"<b>{_pnl_display(profit)}</b>"
            else:
                result_icon = "\u274c"
                profit = t["profit"]
                result_str = f"{_pnl_display(profit)}"

            lines.append(
                f"  {result_icon}  <code>#{tid:>3}</code>  {slot}  "
                f"{dir_icon} <b>{direc:<4}</b>  {prob:.0%}  {result_str}"
            )
        else:
            lines.append(
                f"  \u23f3  <code>#{tid:>3}</code>  {slot}  "
                f"{dir_icon} <b>{direc:<4}</b>  {prob:.0%}  <i>pending...</i>"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    print(format_stats_message())
