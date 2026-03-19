# Neo XG — BTC 5-Min Direction Predictor

AI-powered BTC 5-minute candle direction prediction bot. Predicts whether the next 5-min BTC candle will close UP or DOWN using XGBoost ML, sends signals via Telegram, and supports auto-trading on Polymarket.

---

## 🚀 One-Command Deploy to Railway

**Just set 2 environment variables and deploy — no manual setup needed.**

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new?template=https://github.com/blinkinfo/neoxg)

Or connect your GitHub repo at [railway.app](https://railway.app).

**Required environment variables** (in Railway dashboard → Variables):
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Start command:**
```bash
cd neoxg && pip install -r requirements.txt && python -m src.telegram_bot
```

**That's it.** On first deploy, the bot automatically:
1. Fetches ~90 days of BTC 5-min candles from MEXC
2. Trains the XGBoost model (~1-2 minutes)
3. Starts sending Telegram signals every 5 minutes

---

## 🏗️ Architecture

```
MEXC  (5m candles)
       ↓
Data Fetcher → Feature Engineer → XGBoost Predictor
                                           ↓
                                  Telegram Bot (signals)
                                           ↓
                                  Polymarket (auto-trade)
```

---

## 📁 Project Structure

```
neoxg/
├── src/
│   ├── __init__.py
│   ├── config.py          # All settings
│   ├── provision.py       # Auto-fetch + auto-train on startup
│   ├── data_fetcher.py   # MEXC candle fetching
│   ├── features.py       # RSI, MACD, BB, momentum, ATR
│   ├── trainer.py        # XGBoost training + backtesting
│   ├── predictor.py      # Live prediction engine
│   ├── tracker.py        # Win/loss tracking
│   └── telegram_bot.py   # Telegram bot with auto-signals
├── scripts/
│   ├── fetch_data.sh     # Fetch historical candles
│   ├── train_model.sh    # Train XGBoost model
│   └── run_bot.sh        # Launch Telegram bot
├── tests/
│   └── test_features.py
├── data/                 # Candle data (gitignored)
├── models/               # Saved models (gitignored)
├── logs/                 # Logs (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── Railway.md
```

---

## 💻 Local Development

### 1. Clone & Install

```bash
git clone https://github.com/blinkinfo/neoxg.git
cd neoxg
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Telegram bot token and chat ID
```

### 3. Run

```bash
python -m src.telegram_bot
```

On first run, it auto-fetches candles and trains the model automatically.

---

## 📊 Model Details

- **Algorithm:** XGBoost binary classifier
- **Features (30):** RSI, MACD, Bollinger Bands, momentum, volume ratio, ATR, candlestick patterns, time features
- **Training data:** ~60 days of 5-min BTC/USDT candles from MEXC (auto-provisioned)
- **Validation:** 30 days held out
- **Expected accuracy:** ~52% (profitable with 96¢ payout)
- **Expected value:** ~$0.016 per $1 trade

---

## 🔧 Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/signal` | Get live prediction now |
| `/stats` | Win/loss tracker, streaks, P&L |
| `/status` | Model accuracy stats |
| `/accuracy` | Detailed accuracy report |
| `/help` | Help |

Auto-signals fire every 5 minutes with the next candle's direction.

---

## ⚙️ Configuration

All settings in `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADE_AMOUNT` | `$1` | Amount per trade |
| `PREDICTION_THRESHOLD` | `0.52` | Min probability to signal UP |

---

## 🚂 Railway Deployment

See [Railway.md](Railway.md) for full deployment details.

---

## 📌 Disclaimer

This bot is for educational purposes. Past performance does not guarantee future results. Cryptocurrency trading is risky — trade responsibly.
