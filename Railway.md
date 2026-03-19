# Deploy on Railway

Railway is the easiest way to deploy this bot.

## One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new?template=https://github.com/blinkinfo/neoxg)

Or connect your GitHub repo directly at [railway.app](https://railway.app).

## Environment Variables

Set these in Railway's dashboard → Variables:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELELEGRAM_CHAT_ID=your_chat_id
POLYMARKET_PRIVATE_KEY=your_wallet_private_key   # optional
POLYMARKET_API_KEY=your_api_key                  # optional
POLYMARKET_API_SECRET=your_api_secret            # optional
```

## Start Command

```bash
cd neoxg && pip install -r requirements.txt && python -m src.telegram_bot
```

Or use the provided shell scripts:

```bash
bash scripts/run_bot.sh
```

## Persist Data

Railway's filesystem is ephemeral — use a persistent volume for:
- `data/` — candle CSV data
- `models/` — trained XGBoost model
- `logs/` — bot logs

Mount these at `/data` in Railway dashboard → Volumes.

## Retrain Model

To retrain with fresh data on Railway:

```bash
pip install -r requirements.txt
python -m src.data_fetcher
python -m src.trainer
```

Or use the convenience scripts:
```bash
bash scripts/fetch_data.sh
bash scripts/train_model.sh
```

## Cron Jobs

For periodic retraining, add a Railway cron job:
```bash
0 */6 * * * cd neoxg && python -m src.data_fetcher && python -m src.trainer
```

## Health Check

The bot writes its PID to `logs/bot.pid` on startup.
Add a health check endpoint or use Railway's native checks.

## Troubleshooting

- **Bot not responding?** Check `logs/telegram_bot.log`
- **No candles fetched?** MEXC API may be blocked — try Binance source
- **Model not found?** Run `python -m src.trainer` first
- **Permission errors?** Ensure `logs/` and `data/` are writable
