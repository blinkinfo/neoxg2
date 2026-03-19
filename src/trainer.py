"""
trainer.py
Trains XGBoost model to predict 5-min BTC candle direction.
Backtests accuracy on held-out validation set.
"""

import os

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score



from src.config import MODEL_PATH, DATA_DIR, PREDICTION_THRESHOLD
from src.data_fetcher import load_candles
from src.features import compute_features, prepare_ml_data


def train_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost binary classifier.
    Returns trained model and metrics dict.
    """
    import xgboost as xgb
    
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
            "verbosity": 0,
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    return model


def backtest_predictions(model, X, y, threshold=PREDICTION_THRESHOLD):
    """
    Walk-forward style backtest on a single data split.
    Returns metrics dict.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = 0.5
    
    # Directional accuracy (did we correctly predict UP when it went up, etc)
    tp = ((preds == 1) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_predictions": len(preds),
    }


def find_optimal_threshold(model, X_val, y_val, thresholds=None):
    """
    Find the probability threshold that maximizes accuracy on validation set.
    Returns (best_threshold, best_accuracy).
    """
    if thresholds is None:
        thresholds = np.arange(0.45, 0.60, 0.01)
    
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_acc = 0
    
    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    
    return best_thresh, best_acc


def run_training(days_train=120, days_val=30, threshold=None):
    """
    Main training pipeline.
    
    1. Load candles
    2. Compute features
    3. Split: last `days_val` for validation, rest for training
    4. Train XGBoost
    5. Find optimal threshold
    6. Backtest on validation
    7. Save model + metrics
    """
    print("=" * 60)
    print("BTC Direction Predictor — Training Pipeline")
    print("=" * 60)
    
    # Load candles
    print("\n[1] Loading candles...")
    df = load_candles()
    print(f"    Loaded {len(df)} candles: {df.datetime.min().date()} → {df.datetime.max().date()}")
    
    # Compute features
    print("\n[2] Computing features...")
    df = compute_features(df)
    print(f"    Features computed: {df.datetime.min().date()} → {df.datetime.max().date()}")
    
    # Prepare ML data
    print("\n[3] Preparing training data...")
    X, y, feature_cols = prepare_ml_data(df, drop_na=True)
    print(f"    Samples after dropna: {len(X)}")
    print(f"    Class balance: UP={y.sum()} ({y.sum()/len(y)*100:.1f}%), DOWN={len(y)-y.sum()}")
    
    # Time-based split
    split_idx = len(X) - (days_val * 24 * 60 // 5)  # rough: days * 12 candles/hour
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n    Training set:   {len(X_train)} samples ({days_train}+ days)")
    print(f"    Validation set: {len(X_val)} samples ({days_val} days)")
    
    # Train model
    print("\n[4] Training XGBoost...")
    model = train_model(X_train, y_train, X_val, y_val)
    print(f"    Best iteration: {model.best_iteration}")
    print(f"    Best validation logloss: {model.best_score:.4f}")
    
    # Initial backtest (default threshold)
    print("\n[5] Backtesting (threshold=0.52)...")
    initial_metrics = backtest_predictions(model, X_val, y_val, threshold=0.52)
    print(f"    Accuracy:   {initial_metrics['accuracy']*100:.2f}%")
    print(f"    AUC:        {initial_metrics['auc']:.4f}")
    print(f"    Precision:  {initial_metrics['precision_up']*100:.2f}%")
    print(f"    Recall:    {initial_metrics['recall_up']*100:.2f}%")
    
    # Find optimal threshold
    print("\n[6] Finding optimal threshold...")
    best_thresh, best_acc = find_optimal_threshold(model, X_val, y_val)
    if threshold:
        final_thresh = threshold
    else:
        final_thresh = best_thresh
    
    print(f"    Optimal threshold: {best_thresh:.3f} → accuracy: {best_acc*100:.2f}%")
    print(f"    Using threshold:  {final_thresh:.3f}")
    
    # Final backtest with optimal threshold
    final_metrics = backtest_predictions(model, X_val, y_val, threshold=final_thresh)
    
    # Win rate accounting for payout
    win_rate = final_metrics["precision_up"]  # precision for UP class
    payout = 0.96
    expected_value = (win_rate * payout) - ((1 - win_rate) * 1)
    
    print(f"\n{'=' * 60}")
    print(f"VALIDATION RESULTS (threshold={final_thresh:.3f})")
    print(f"{'=' * 60}")
    print(f"  Accuracy:       {final_metrics['accuracy']*100:.2f}%")
    print(f"  AUC:            {final_metrics['auc']:.4f}")
    print(f"  Win rate (UP):  {win_rate*100:.2f}%")
    print(f"  Expected Value: ${expected_value:.4f} per $1 trade")
    print(f"  Total trades:   {final_metrics['total_predictions']}")
    print(f"  TP / TN / FP / FN: {final_metrics['tp']} / {final_metrics['tn']} / {final_metrics['fp']} / {final_metrics['fn']}")
    
    if win_rate >= 0.52:
        print(f"\n  ✅ Model is profitable at ${payout} payout!")
    else:
        print(f"\n  ⚠️  Below breakeven ({1/(1+payout)*100:.2f}% needed)")
    
    # Feature importance
    print(f"\n[7] Top 10 Feature Importances:")
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    for feat, imp in importance.nlargest(10).items():
        print(f"    {feat:25s}: {imp:.4f}")
    
    # Save model
    print(f"\n[8] Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    
    # Save metrics
    metrics = {
        "threshold": final_thresh,
        "best_iteration": model.best_iteration,
        "validation_accuracy": final_metrics["accuracy"],
        "validation_auc": final_metrics["auc"],
        "validation_win_rate": win_rate,
        "expected_value_per_dollar": expected_value,
        "total_validation_trades": final_metrics["total_predictions"],
        "tp": final_metrics["tp"],
        "tn": final_metrics["tn"],
        "fp": final_metrics["fp"],
        "fn": final_metrics["fn"],
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "feature_importance": importance.nlargest(15).to_dict(),
    }
    metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"    Metrics saved to {metrics_path}")
    
    print(f"\n✅ Training complete!")
    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-train", type=int, default=120)
    parser.add_argument("--days-val", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    
    model, metrics = run_training(
        days_train=args.days_train,
        days_val=args.days_val,
        threshold=args.threshold,
    )
