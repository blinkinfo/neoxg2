# Deploy on Railway

## One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new?template=https://github.com/blinkinfo/neoxg)

Or connect your GitHub repo at [railway.app](https://railway.app).

## Required Environment Variables

Set these in Railway dashboard → Variables:

| Variable | Value |
|----------|-------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID from @userinfobot |

**No other variables needed.** The bot auto-provisions its own data and model on first startup.

## Start Command

```bash
pip install -r requirements.txt && python -m src.telegram_bot
```

## What Happens on First Deploy

1. Railway starts the bot
2. Bot detects no data/model → auto-fetches ~200 days of BTC candles from MEXC (~40 seconds)
3. Auto-trains XGBoost + LightGBM ensemble model (~2-3 minutes)
4. Starts sending Telegram signals every 5 minutes

After that, it reads existing data — starts in seconds.

## Filesystem Notes

Railway's filesystem is ephemeral. Data/model are stored in `data/` and `models/` within the project. On restart:
- **Without a volume:** Data re-provisions automatically (candles + training ~2 min)
- **With a volume:** Data persists across restarts (faster startup)

To add a persistent volume in Railway:
1. Create a volume in Railway dashboard
2. Mount it at `/data/btc-predictor` (the project root)

## Retraining

To retrain with fresh data, use a Railway cron job or redeploy:

```bash
python -m src.data_fetcher
python -m src.trainer
```

Or via the convenience scripts:
```bash
bash scripts/fetch_data.sh
bash scripts/train_model.sh
```

## Troubleshooting

- **Bot not responding?** Check `logs/telegram_bot.log`
- **Model errors?** Ensure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set correctly
- **API errors?** MEXC may be temporarily blocked — bot retries automatically
