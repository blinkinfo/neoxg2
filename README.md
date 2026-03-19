# Neo XG — BTC 5-Min Direction Predictor

AI-powered BTC 5-minute candle direction prediction bot. Predicts whether the next 5-min BTC candle will close UP or DOWN using XGBoost ML, sends signals via Telegram, and supports auto-trading on Polymarket.

---

## 🏗️ Architecture

```
Binance / MEXC  (5m candles)
       ↓
Data Fetcher  →  Feature Engineer  →  XGBoost Predictor
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
│   ├── config.py          # All settings — edit this
│   ├── data_fetcher.py    # MEXC candle fetching
│   ├── features.py         # RSI, MACD, BB, momentum, ATR
│   ├── trainer.py          # XGBoost training + backtesting
│   ├── predictor.py        # Live prediction engine
│   ├── tracker.py          # Win/loss tracking
│   └── telegram_bot.py     # Telegram bot with auto-signals
├── scripts/
│   ├── fetch_data.sh       # Fetch historical candles
│   ├── train_model.sh      # Train XGBoost model
│   └── run_bot.sh          # Launch Telegram bot
├── tests/
│   └── test_features.py    # Feature engineering tests
├── data/                   # Candle data + model output (gitignored)
├── models/                 # Saved models (gitignored)
├── logs/                   # Logs (gitignored)
├── requirements.txt
├── .env.example            # Template for secrets
├── .gitignore
├── README.md
└── Railway.md              # Railway deployment guide
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/blinkinfo/neoxg.git
cd neoxg
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your secrets
```

**Required variables in `.env`:**
```env
# Telegram Bot (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Polymarket (for auto-trading — optional)
POLYMARKET_PRIVATE_KEY=your_wallet_private_key
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_api_secret
```

### 3. Fetch Data & Train

```bash
python -m src.data_fetcher   # Fetch 180 days of BTC candles
python -m src.trainer         # Train XGBoost model
```

### 4. Run

```bash
python -m src.telegram_bot    # Start Telegram bot
```

---

## 📊 Model Details

- **Algorithm:** XGBoost binary classifier
- **Features (30):** RSI, MACD, Bollinger Bands, momentum, volume ratio, ATR, candlestick patterns, time features
- **Training data:** 150 days of 5-min BTC/USDT candles from MEXC
- **Validation:** 30 days held out
- **Validation accuracy:** ~52.4% (profitable with 96¢ payout)
- **Expected value:** ~$0.016 per $1 trade

---

## 🔧 Commands (Telegram Bot)

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
| `RSI_PERIOD` | `14` | RSI lookback |
| `MACD_FAST/SLOW/SIGNAL` | `12/26/9` | MACD parameters |
| `BB_PERIOD` | `20` | Bollinger Bands period |

---

## 🚂 Deployment (Railway)

See [Railway.md](Railway.md) for one-command Railway deployment.

---

## 📌 Disclaimer

This bot is for educational purposes. Past performance does not guarantee future results. Cryptocurrency trading is risky — trade responsibly.
