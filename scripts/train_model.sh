#!/bin/bash
# Train XGBoost model on fetched BTC candle data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🤖 Training XGBoost model..."
python -m src.trainer --days-train 150 --days-val 30

echo "✅ Training complete. Model saved to models/"
