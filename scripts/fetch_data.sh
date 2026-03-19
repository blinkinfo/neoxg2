#!/bin/bash
# Fetch historical BTC 5-min candles from MEXC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "📡 Fetching 180 days of BTC 5-min candles from MEXC..."
python -m src.data_fetcher

echo "✅ Data fetch complete. Saved to data/btc_candles.csv"
