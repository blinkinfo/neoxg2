"""
trainer.py
Trains XGBoost + LightGBM ensemble to predict 5-min BTC candle direction.
Uses walk-forward cross-validation for robust evaluation.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score

from src.config import (
    MODEL_PATH, LIGHTGBM_MODEL_PATH, DATA_DIR,
    PREDICTION_THRESHOLD, ENSEMBLE_WEIGHTS
)
from src.data_fetcher import load_candles
from src.features import compute_features, prepare_ml_data


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost binary classifier."""
    import xgboost as xgb
    
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 5,
            "gamma": 0.2,
            "reg_alpha": 0.3,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_estimators": 500,
            "early_stopping_rounds": 40,
            "verbosity": 0,
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM binary classifier."""
    import lightgbm as lgb
    
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_samples": 20,
            "reg_alpha": 0.3,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_estimators": 500,
            "verbose": -1,
        }
    
    callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    
    return model


def ensemble_predict_proba(xgb_model, lgb_model, X, weights=None):
    """
    Get ensemble probability from both models.
    Returns probability of UP (class 1).
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS
    
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    lgb_proba = lgb_model.predict_proba(X)[:, 1]
    
    ensemble_proba = (
        weights["xgboost"] * xgb_proba +
        weights["lightgbm"] * lgb_proba
    )
    
    return ensemble_proba


def backtest_predictions(proba, y, threshold=PREDICTION_THRESHOLD):
    """
    Compute accuracy metrics from probability predictions.
    Returns metrics dict.
    """
    preds = (proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = 0.5
    
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


def walk_forward_cv(X, y, n_splits=5):
    """
    Walk-forward cross-validation for time series.
    Returns list of (train_idx, val_idx) tuples.
    Each fold uses all data before the fold as training,
    and a fixed window as validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    return splits


def find_optimal_threshold(proba, y, thresholds=None):
    """
    Find the probability threshold that maximizes accuracy.
    Uses all walk-forward validation predictions combined.
    Returns (best_threshold, best_accuracy).
    """
    if thresholds is None:
        thresholds = np.arange(0.45, 0.60, 0.005)
    
    best_thresh = 0.5
    best_acc = 0
    
    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    
    return best_thresh, best_acc


def run_training(days_train=120, days_val=30, threshold=None):
    """
    Main training pipeline with walk-forward CV and ensemble.
    
    1. Load candles
    2. Compute features
    3. Walk-forward cross-validation (5 folds)
    4. Train XGBoost + LightGBM ensemble
    5. Find optimal threshold across all folds
    6. Final train on all data except last `days_val` for final eval
    7. Save models + metrics
    """
    print("=" * 60)
    print("BTC Direction Predictor -- Enhanced Training Pipeline")
    print("=" * 60)
    
    # Load candles
    print("\n[1] Loading candles...")
    df = load_candles()
    print(f"    Loaded {len(df)} candles: {df.datetime.min().date()} -> {df.datetime.max().date()}")
    
    # Compute features
    print("\n[2] Computing features...")
    df = compute_features(df)
    print(f"    Features computed: {df.datetime.min().date()} -> {df.datetime.max().date()}")
    
    # Prepare ML data (flat candles filtered out)
    print("\n[3] Preparing training data...")
    X, y, feature_cols = prepare_ml_data(df, drop_na=True)
    print(f"    Samples after dropna + flat filter: {len(X)}")
    print(f"    Features: {len(feature_cols)}")
    print(f"    Class balance: UP={y.sum()} ({y.sum()/len(y)*100:.1f}%), DOWN={len(y)-y.sum()}")
    
    # Walk-forward cross-validation
    print(f"\n[4] Walk-forward cross-validation (5 folds)...")
    splits = walk_forward_cv(X, y, n_splits=5)
    
    all_val_proba = []
    all_val_y = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train both models on this fold
        xgb_model = train_xgboost(X_tr, y_tr, X_vl, y_vl)
        lgb_model = train_lightgbm(X_tr, y_tr, X_vl, y_vl)
        
        # Ensemble predictions
        fold_proba = ensemble_predict_proba(xgb_model, lgb_model, X_vl)
        fold_acc = accuracy_score(y_vl, (fold_proba >= 0.5).astype(int))
        
        all_val_proba.extend(fold_proba)
        all_val_y.extend(y_vl.values)
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "train_size": len(X_tr),
            "val_size": len(X_vl),
            "accuracy": fold_acc,
        })
        
        print(f"    Fold {fold_idx+1}: train={len(X_tr)}, val={len(X_vl)}, acc={fold_acc:.4f}")
    
    all_val_proba = np.array(all_val_proba)
    all_val_y = np.array(all_val_y)
    
    cv_accuracy = accuracy_score(all_val_y, (all_val_proba >= 0.5).astype(int))
    print(f"\n    Mean CV Accuracy: {cv_accuracy:.4f}")
    
    # Find optimal threshold across all CV predictions
    print("\n[5] Finding optimal threshold across all folds...")
    best_thresh, best_acc = find_optimal_threshold(all_val_proba, all_val_y)
    if threshold:
        final_thresh = threshold
    else:
        final_thresh = best_thresh
    
    print(f"    Optimal threshold: {best_thresh:.3f} -> accuracy: {best_acc*100:.2f}%")
    print(f"    Using threshold:  {final_thresh:.3f}")
    
    # Final training: use all data except last days_val for final eval
    print("\n[6] Training final models...")
    split_idx = len(X) - (days_val * 24 * 60 // 5)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"    Training set:   {len(X_train)} samples")
    print(f"    Validation set: {len(X_val)} samples")
    
    xgb_final = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_final = train_lightgbm(X_train, y_train, X_val, y_val)
    
    print(f"    XGBoost best iteration: {xgb_final.best_iteration}")
    
    # Final eval with ensemble
    final_proba = ensemble_predict_proba(xgb_final, lgb_final, X_val)
    final_metrics = backtest_predictions(final_proba, y_val, threshold=final_thresh)
    
    # Win rate and EV
    win_rate = final_metrics["accuracy"]
    payout = 0.96
    expected_value = (win_rate * payout) - ((1 - win_rate) * 1)
    
    print(f"\n{'=' * 60}")
    print(f"VALIDATION RESULTS (threshold={final_thresh:.3f})")
    print(f"{'=' * 60}")
    print(f"  Accuracy:       {final_metrics['accuracy']*100:.2f}%")
    print(f"  AUC:            {final_metrics['auc']:.4f}")
    print(f"  Win rate:       {win_rate*100:.2f}%")
    print(f"  Expected Value: ${expected_value:.4f} per $1 trade")
    print(f"  Total trades:   {final_metrics['total_predictions']}")
    print(f"  TP / TN / FP / FN: {final_metrics['tp']} / {final_metrics['tn']} / {final_metrics['fp']} / {final_metrics['fn']}")
    
    if win_rate >= 0.52:
        print(f"\n  OK Model is profitable at ${payout} payout!")
    else:
        print(f"\n  WARNING Below breakeven ({1/(1+payout)*100:.2f}% needed)")
    
    # Feature importance (XGBoost)
    print(f"\n[7] Top 10 Feature Importances (XGBoost):")
    importance = pd.Series(xgb_final.feature_importances_, index=feature_cols)
    for feat, imp in importance.nlargest(10).items():
        print(f"    {feat:25s}: {imp:.4f}")
    
    # Save models
    print(f"\n[8] Saving models...")
    xgb_final.save_model(str(MODEL_PATH))
    lgb_final.booster_.save_model(str(LIGHTGBM_MODEL_PATH))
    print(f"    XGBoost saved to {MODEL_PATH}")
    print(f"    LightGBM saved to {LIGHTGBM_MODEL_PATH}")
    
    # Save metrics
    metrics = {
        "threshold": final_thresh,
        "best_iteration": xgb_final.best_iteration,
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
        "feature_cols": feature_cols,
        "cv_accuracy": cv_accuracy,
        "cv_folds": fold_metrics,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "feature_importance": importance.nlargest(15).to_dict(),
    }
    metrics_path = str(MODEL_PATH).replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"    Metrics saved to {metrics_path}")
    
    print(f"\n Training complete!")
    return xgb_final, metrics


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
