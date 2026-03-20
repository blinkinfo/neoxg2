"""
config.py — All settings for the BTC Predictor bot.
Secrets are loaded from environment variables.
Copy .env.example to .env and fill in values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
dotenv_path = Path(__file__).parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ── Project paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# ── Data source ─────────────────────────────────────────────────
MEXC_SYMBOL = "BTC/USDT"
MEXC_INTERVAL = "5m"
CANDLE_LIMIT = 1000  # max per request

# ── Feature config ──────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
MOMENTUM_PERIODS = [3, 5, 7]
VOL_LOOKBACK = 20

# ── Order book config ───────────────────────────────────────────
ORDERBOOK_DEPTH = 20  # levels to fetch for bid/ask imbalance

# ── Funding rate config ─────────────────────────────────────────
FUNDING_RATE_SYMBOL = "BTC/USDT:USDT"  # MEXC perpetual swap symbol

# ── Regime filter config ────────────────────────────────────────
VOLATILITY_REGIME_LOOKBACK = 50  # candles to compute volatility regime
LOW_VOLATILITY_ATR_PERCENTILE = 20  # below this = low vol regime

# ── Ensemble config ─────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {"xgboost": 0.5, "lightgbm": 0.5}  # equal weight by default
LIGHTGBM_MODEL_PATH = MODELS_DIR / "btc_direction_lgb.txt"

# ── Confidence filter config ────────────────────────────────────
MIN_CONFIDENCE_TO_TRADE = float(os.getenv("MIN_CONFIDENCE_TO_TRADE", "0.10"))
HIGH_CONFIDENCE_THRESHOLD = 0.40

# ── Model paths ─────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "btc_direction_model.json"
MODEL_BACKUP_PATH = MODELS_DIR / "btc_direction_model_backup.json"
LIGHTGBM_BACKUP_PATH = MODELS_DIR / "btc_direction_lgb_backup.txt"

# ── Prediction config ───────────────────────────────────────────
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.52"))

# ── Auto-retrain config ────────────────────────────────────────
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
RETRAIN_MAX_CONSECUTIVE_REJECTIONS = int(os.getenv("RETRAIN_MAX_REJECTIONS", "5"))

# ── Telegram (required) ────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ── Polymarket (optional — for auto-trading) ───────────────────
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "").strip()
POLYMARKET_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "").strip()
POLYMARKET_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "").strip()

# ── Trading config ──────────────────────────────────────────────
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "1.0"))  # $ per trade
PAYOUT = 0.96  # win payout

# ── Validation ──────────────────────────────────────────────────
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set. Copy .env.example to .env and fill in your Telegram bot token.")
if not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_CHAT_ID not set. Copy .env.example to .env and fill in your chat ID.")
