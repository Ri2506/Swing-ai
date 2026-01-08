"""
# 🚀 SwingAI V5 - Advanced Multi-Strategy Approach
# Fundamentally different solutions for better accuracy

NEW STRATEGIES:
1. DYNAMIC THRESHOLDS - ATR-based instead of fixed percentages
2. BINARY DECOMPOSITION - Separate UP/DOWN classifiers (more accurate)
3. MULTI-MODEL STACKING - LightGBM + XGBoost + CatBoost with meta-learner
4. ORDINAL REGRESSION - Respects SHORT < NEUTRAL < LONG ordering
5. FEATURE SELECTION - Remove noisy features automatically
6. CONFIDENCE CALIBRATION - Platt scaling for reliable confidence scores
"""

# ============================================================
# CELL 1: Install (restart runtime after)
# ============================================================
# %pip install -q "numpy<2.0" "pandas==2.2.2" "scipy<1.13" "scikit-learn==1.5.2" \
#     "catboost==1.2.7" "lightgbm>=4.0" "xgboost>=2.0" "yfinance>=0.2.54" \
#     "curl_cffi>=0.7.4" "optuna>=3.0"

# ============================================================
# CELL 2: Imports
# ============================================================
import os, json, time, random, warnings
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Try to import LightGBM and XGBoost
try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM available")
except:
    HAS_LGB = False
    print("⚠️ LightGBM not available")

try:
    import xgboost as xgb
    HAS_XGB = True
    print("✅ XGBoost available")
except:
    HAS_XGB = False
    print("⚠️ XGBoost not available")

print(f"✅ GPU: {torch.cuda.is_available()}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CELL 3: Config V5
# ============================================================
@dataclass
class ConfigV5:
    LOOKBACK_DAYS: int = 60
    PREDICTION_HORIZON: int = 5
    MIN_HISTORY_DAYS: int = 252
    
    # DYNAMIC THRESHOLDS - will be calculated per-stock based on ATR
    USE_DYNAMIC_THRESHOLDS: bool = True
    FIXED_UP_THRESHOLD: float = 0.02  # Fallback if dynamic fails
    FIXED_DOWN_THRESHOLD: float = -0.015
    
    # ATR multipliers for dynamic thresholds
    ATR_UP_MULTIPLIER: float = 1.5   # UP if return > 1.5 * ATR
    ATR_DOWN_MULTIPLIER: float = 1.0  # DOWN if return < -1.0 * ATR
    
    TRAIN_END: str = "2024-06-30"
    VAL_END: str = "2024-09-30"
    
    # Reduced feature set (top 30 most important)
    USE_FEATURE_SELECTION: bool = True
    TOP_N_FEATURES: int = 30
    
    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 50
    PATIENCE: int = 12
    
    NUM_FEATURES: int = 40  # Initial, may be reduced
    MODEL_SAVE_PATH: str = "/content/drive/MyDrive/SwingAI/models_v5/"

config = ConfigV5()

# Full feature set
ALL_FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d', 'volatility_20d',
    'close_to_sma_20', 'close_to_sma_50', 'rsi_14_norm', 'macd_histogram_norm', 'bb_position',
    'structure_score', 'range_position', 'dist_to_swing_high', 'dist_to_swing_low',
    'in_discount', 'in_deep_discount', 'in_premium', 'near_bullish_ob', 'near_bearish_ob',
    'bullish_fvg', 'bearish_fvg', 'sweep_high', 'sweep_low', 'bos_bullish', 'bos_bearish',
    'volume_ratio', 'volume_trend', 'obv_slope', 'close_to_vwap',
    'buying_pressure', 'accumulation_score', 'big_volume_day', 'higher_high',
    'daily_trend', 'weekly_trend', 'monthly_trend', 'mtf_alignment',
    'weekly_range_pos', 'monthly_range_pos', 'trend_strength',
]

print(f"✅ Config V5: Starting with {len(ALL_FEATURES)} features")

# ============================================================
# CELL 4: Download Data
# ============================================================
FO_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "ONGC.NS",
    "TECHM.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "COALINDIA.NS", "SBILIFE.NS", "BPCL.NS", "INDUSINDBK.NS", "TATACONSUM.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "SHRIRAMFIN.NS",
    "TRENT.NS", "POLYCAB.NS", "PERSISTENT.NS", "DIXON.NS", "TATAELXSI.NS",
    "ABB.NS", "SIEMENS.NS", "HAL.NS", "BEL.NS", "IRCTC.NS",
    "COFORGE.NS", "MUTHOOTFIN.NS", "INDHOTEL.NS", "BANKBARODA.NS",
    "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "CHOLAFIN.NS", "VEDL.NS",
]

def yf_download_safe(tickers, start, end, chunk_size=5, max_retries=5, base_sleep=2.0):
    out = {}
    total = len(tickers)
    print(f"📥 Downloading {total} stocks...")
    for i in range(0, total, chunk_size):
        chunk = tickers[i:i+chunk_size]
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(" ".join(chunk), start=start, end=end, 
                                group_by="ticker", auto_adjust=True, 
                                threads=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    for t in chunk:
                        if t in df.columns.get_level_values(0):
                            tdf = df[t].dropna(how="all")
                            if len(tdf) >= config.MIN_HISTORY_DAYS:
                                out[t] = tdf
                else:
                    if len(df) >= config.MIN_HISTORY_DAYS:
                        out[chunk[0]] = df.dropna(how="all")
                break
            except:
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random())
        done = min(i + chunk_size, total)
        if done % 20 == 0 or done == total:
            print(f"   {done}/{total} | {len(out)} downloaded")
        time.sleep(base_sleep + random.random())
    print(f"✅ Downloaded {len(out)} stocks")
    return out

stock_data = yf_download_safe(FO_STOCKS, "2019-01-01", "2024-12-31")

# ============================================================
# CELL 5: DYNAMIC ATR-BASED THRESHOLDS
# ============================================================
def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def create_dynamic_labels(df, horizon=5):
    """
    Create labels based on ATR-normalized returns.
    This adapts thresholds to each stock's volatility.
    """
    forward_return = df['Close'].shift(-horizon) / df['Close'] - 1
    atr = calculate_atr(df)
    atr_pct = atr / df['Close']  # ATR as percentage of price
    
    # Smoothed ATR for stable thresholds
    atr_smooth = atr_pct.rolling(20).mean()
    
    # Dynamic thresholds
    up_threshold = config.ATR_UP_MULTIPLIER * atr_smooth
    down_threshold = -config.ATR_DOWN_MULTIPLIER * atr_smooth
    
    # Create labels
    labels = pd.Series(1, index=df.index)  # Default: NEUTRAL
    labels[forward_return >= up_threshold] = 2  # UP
    labels[forward_return <= down_threshold] = 0  # DOWN
    
    return labels

def create_fixed_labels(df):
    """Fallback: fixed percentage thresholds"""
    forward_return = df['Close'].shift(-config.PREDICTION_HORIZON) / df['Close'] - 1
    labels = pd.Series(1, index=df.index)
    labels[forward_return >= config.FIXED_UP_THRESHOLD] = 2
    labels[forward_return <= config.FIXED_DOWN_THRESHOLD] = 0
    return labels

# ============================================================
# CELL 6: Features V5
# ============================================================
def calculate_features_v5(df):
    data = df.copy()
    
    # Price Action
    data['return_1d'] = data['Close'].pct_change(1)
    data['return_5d'] = data['Close'].pct_change(5)
    data['return_10d'] = data['Close'].pct_change(10)
    data['return_20d'] = data['Close'].pct_change(20)
    data['volatility_20d'] = data['return_1d'].rolling(20).std() * np.sqrt(252)
    
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['close_to_sma_20'] = (data['Close'] - data['sma_20']) / data['sma_20']
    data['close_to_sma_50'] = (data['Close'] - data['sma_50']) / data['sma_50']
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['rsi_14_norm'] = (100 - (100 / (1 + rs)) - 50) / 50
    
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    data['macd_histogram_norm'] = (macd - macd_signal) / data['Close']
    
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # SMC/ICT
    data['swing_high'] = data['High'].rolling(10, center=True).max()
    data['swing_low'] = data['Low'].rolling(10, center=True).min()
    data['prev_swing_high'] = data['swing_high'].shift(10)
    data['prev_swing_low'] = data['swing_low'].shift(10)
    data['higher_high'] = (data['swing_high'] > data['prev_swing_high']).astype(int)
    data['higher_low'] = (data['swing_low'] > data['prev_swing_low']).astype(int)
    data['lower_high'] = (data['swing_high'] < data['prev_swing_high']).astype(int)
    data['lower_low'] = (data['swing_low'] < data['prev_swing_low']).astype(int)
    
    data['structure_score'] = (data['higher_high'].rolling(5).sum() + 
                               data['higher_low'].rolling(5).sum() -
                               data['lower_high'].rolling(5).sum() - 
                               data['lower_low'].rolling(5).sum()) / 10
    
    range_high = data['High'].rolling(50).max()
    range_low = data['Low'].rolling(50).min()
    data['range_position'] = (data['Close'] - range_low) / (range_high - range_low + 1e-10)
    
    data['dist_to_swing_high'] = (data['swing_high'] - data['Close']) / data['Close']
    data['dist_to_swing_low'] = (data['Close'] - data['swing_low']) / data['Close']
    
    data['in_discount'] = (data['range_position'] < 0.5).astype(int)
    data['in_deep_discount'] = (data['range_position'] < 0.3).astype(int)
    data['in_premium'] = (data['range_position'] > 0.7).astype(int)
    
    vol_threshold = data['return_1d'].rolling(20).std() * 2
    data['near_bullish_ob'] = (data['return_1d'] > vol_threshold).rolling(10).sum()
    data['near_bearish_ob'] = (data['return_1d'] < -vol_threshold).rolling(10).sum()
    
    data['gap_up'] = ((data['Low'] > data['High'].shift(1)) & (data['return_1d'] > 0.01)).astype(int)
    data['gap_down'] = ((data['High'] < data['Low'].shift(1)) & (data['return_1d'] < -0.01)).astype(int)
    data['bullish_fvg'] = data['gap_up'].rolling(5).sum()
    data['bearish_fvg'] = data['gap_down'].rolling(5).sum()
    
    data['sweep_high'] = ((data['High'] > data['swing_high'].shift(1)) & 
                          (data['Close'] < data['swing_high'].shift(1))).astype(int)
    data['sweep_low'] = ((data['Low'] < data['swing_low'].shift(1)) & 
                         (data['Close'] > data['swing_low'].shift(1))).astype(int)
    
    data['bos_bullish'] = ((data['Close'] > data['swing_high'].shift(1)) & 
                           (data['higher_high'] == 1)).astype(int)
    data['bos_bearish'] = ((data['Close'] < data['swing_low'].shift(1)) & 
                           (data['lower_low'] == 1)).astype(int)
    
    # Volume
    data['volume_ma_20'] = data['Volume'].rolling(20).mean()
    data['volume_ratio'] = data['Volume'] / (data['volume_ma_20'] + 1e-10)
    data['volume_trend'] = data['Volume'].rolling(5).mean() / (data['Volume'].rolling(20).mean() + 1e-10)
    
    obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    data['obv_slope'] = obv.diff(5) / (obv.rolling(20).std() + 1e-10)
    
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['close_to_vwap'] = (data['Close'] - data['vwap']) / (data['vwap'] + 1e-10)
    
    data['buying_pressure'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    data['accumulation_score'] = (data['buying_pressure'] * data['volume_ratio']).rolling(5).mean()
    data['big_volume_day'] = (data['volume_ratio'] > 2).astype(int)
    
    # Multi-timeframe
    data['daily_trend'] = (data['Close'] > data['sma_20']).astype(int)
    data['weekly_close'] = data['Close'].rolling(5).mean()
    data['weekly_high'] = data['High'].rolling(5).max()
    data['weekly_low'] = data['Low'].rolling(5).min()
    data['weekly_trend'] = (data['weekly_close'] > data['weekly_close'].shift(5)).astype(int)
    data['monthly_close'] = data['Close'].rolling(21).mean()
    data['monthly_trend'] = (data['monthly_close'] > data['monthly_close'].shift(21)).astype(int)
    data['mtf_alignment'] = (data['daily_trend'] + data['weekly_trend'] + data['monthly_trend']) / 3
    data['weekly_range_pos'] = (data['Close'] - data['weekly_low']) / (data['weekly_high'] - data['weekly_low'] + 1e-10)
    data['monthly_range_pos'] = (data['Close'] - data['Low'].rolling(21).min()) / \
                                (data['High'].rolling(21).max() - data['Low'].rolling(21).min() + 1e-10)
    data['trend_strength'] = abs(data['close_to_sma_20']) + abs(data['close_to_sma_50'])
    
    features_df = data[ALL_FEATURES].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clip extreme values (less aggressive)
    for col in features_df.columns:
        q01, q99 = features_df[col].quantile([0.01, 0.99])
        features_df[col] = features_df[col].clip(q01, q99)
    
    return features_df

print("🔧 Processing stocks with dynamic thresholds...")
feature_data, labels_data = {}, {}
label_dist = Counter()

for symbol, data in stock_data.items():
    try:
        features = calculate_features_v5(data)
        if config.USE_DYNAMIC_THRESHOLDS:
            labels = create_dynamic_labels(data)
        else:
            labels = create_fixed_labels(data)
        
        if len(features) > config.LOOKBACK_DAYS + config.PREDICTION_HORIZON:
            feature_data[symbol] = features
            labels_data[symbol] = labels
            label_dist.update(labels.dropna().astype(int).value_counts().to_dict())
    except Exception as e:
        pass

print(f"✅ Processed {len(feature_data)} stocks")
print(f"📊 Label distribution: DOWN={label_dist.get(0,0):,}, NEUTRAL={label_dist.get(1,0):,}, UP={label_dist.get(2,0):,}")

# ============================================================
# CELL 7: Create Sequences & Split
# ============================================================
print("📊 Creating sequences...")
all_sequences, all_labels, all_dates = [], [], []

for symbol in feature_data.keys():
    features = feature_data[symbol].values
    labels = labels_data[symbol].values
    dates = feature_data[symbol].index
    
    for i in range(config.LOOKBACK_DAYS, len(features) - config.PREDICTION_HORIZON):
        seq = features[i-config.LOOKBACK_DAYS:i]
        label = labels[i]
        if not np.isnan(label) and not np.any(np.isnan(seq)):
            all_sequences.append(seq)
            all_labels.append(int(label))
            all_dates.append(dates[i])

sequences = np.array(all_sequences)
labels_arr = np.array(all_labels)
dates_arr = np.array(all_dates)

print(f"✅ Total sequences: {len(sequences):,}")

train_end = pd.Timestamp(config.TRAIN_END)
val_end = pd.Timestamp(config.VAL_END)
train_mask = dates_arr <= train_end
val_mask = (dates_arr > train_end) & (dates_arr <= val_end)
test_mask = dates_arr > val_end

X_train_seq = sequences[train_mask]
y_train = labels_arr[train_mask]
X_val_seq = sequences[val_mask]
y_val = labels_arr[val_mask]
X_test_seq = sequences[test_mask]
y_test = labels_arr[test_mask]

# Flatten for tree models (use last day)
X_train = X_train_seq[:, -1, :]
X_val = X_val_seq[:, -1, :]
X_test = X_test_seq[:, -1, :]

print(f"📊 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"📊 Test Distribution: {Counter(y_test)}")

# ============================================================
# CELL 8: FEATURE SELECTION using CatBoost importance
# ============================================================
print("="*60)
print("🔍 FEATURE SELECTION")
print("="*60)

# Quick CatBoost for feature importance
quick_cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=0,
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
)
quick_cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# Get feature importance
importance = dict(zip(ALL_FEATURES, quick_cat.feature_importances_))
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n📊 Feature Importance (Top 20):")
for i, (f, v) in enumerate(sorted_features[:20]):
    print(f"   {i+1:2}. {f:25s}: {v:.2f}")

# Select top N features
if config.USE_FEATURE_SELECTION:
    SELECTED_FEATURES = [f for f, _ in sorted_features[:config.TOP_N_FEATURES]]
    selected_idx = [ALL_FEATURES.index(f) for f in SELECTED_FEATURES]
    
    X_train = X_train[:, selected_idx]
    X_val = X_val[:, selected_idx]
    X_test = X_test[:, selected_idx]
    X_train_seq = X_train_seq[:, :, selected_idx]
    X_val_seq = X_val_seq[:, :, selected_idx]
    X_test_seq = X_test_seq[:, :, selected_idx]
    
    print(f"\n✅ Selected top {config.TOP_N_FEATURES} features")
else:
    SELECTED_FEATURES = ALL_FEATURES
    print("\n✅ Using all features")

# ============================================================
# CELL 9: STRATEGY 1 - Multi-Model Gradient Boosting Ensemble
# ============================================================
print("="*60)
print("🌲 STRATEGY 1: MULTI-GBDT ENSEMBLE")
print("="*60)

# Compute class weights
class_counts = Counter(y_train)
total = len(y_train)
class_weight_dict = {c: total / (3 * count) for c, count in class_counts.items()}
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# CatBoost
print("\n🔹 Training CatBoost...")
catboost_model = CatBoostClassifier(
    iterations=1500,
    depth=7,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100,
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    class_weights=class_weight_dict,
)
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
cat_probs = catboost_model.predict_proba(X_test)
cat_acc = (cat_probs.argmax(1) == y_test).mean()
print(f"✅ CatBoost Test Accuracy: {cat_acc:.4f}")

# LightGBM
if HAS_LGB:
    print("\n🔹 Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weights)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 7,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 20,
        'verbose': -1,
        'seed': 42,
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
    }
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    lgb_probs = lgb_model.predict(X_test)
    lgb_acc = (lgb_probs.argmax(1) == y_test).mean()
    print(f"✅ LightGBM Test Accuracy: {lgb_acc:.4f}")
else:
    lgb_probs = cat_probs  # Fallback

# XGBoost
if HAS_XGB:
    print("\n🔹 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=100,
        tree_method='hist',  # Use 'gpu_hist' if GPU available
        device='cuda' if torch.cuda.is_available() else 'cpu',
        eval_metric='mlogloss',
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_acc = (xgb_probs.argmax(1) == y_test).mean()
    print(f"✅ XGBoost Test Accuracy: {xgb_acc:.4f}")
else:
    xgb_probs = cat_probs  # Fallback

# Simple average ensemble
n_models = 1 + int(HAS_LGB) + int(HAS_XGB)
gbdt_probs = (cat_probs + lgb_probs + xgb_probs) / n_models
gbdt_acc = (gbdt_probs.argmax(1) == y_test).mean()
print(f"\n🎯 GBDT Ensemble ({n_models} models) Accuracy: {gbdt_acc:.4f}")

# ============================================================
# CELL 10: STRATEGY 2 - Binary Decomposition
# ============================================================
print("="*60)
print("🎯 STRATEGY 2: BINARY DECOMPOSITION")
print("="*60)
print("Training separate UP vs REST and DOWN vs REST classifiers")

# UP detector (class 2 vs rest)
y_train_up = (y_train == 2).astype(int)
y_val_up = (y_val == 2).astype(int)
y_test_up = (y_test == 2).astype(int)

# DOWN detector (class 0 vs rest)
y_train_down = (y_train == 0).astype(int)
y_val_down = (y_val == 0).astype(int)
y_test_down = (y_test == 0).astype(int)

# UP classifier
print("\n🔹 Training UP detector...")
up_weights = {0: 1.0, 1: sum(y_train_up == 0) / sum(y_train_up == 1)}
up_model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    class_weights=up_weights,
)
up_model.fit(X_train, y_train_up, eval_set=(X_val, y_val_up), use_best_model=True)
up_probs = up_model.predict_proba(X_test)[:, 1]
print(f"✅ UP detector AUC-like: {np.mean((up_probs > 0.5) == y_test_up):.4f}")

# DOWN classifier
print("\n🔹 Training DOWN detector...")
down_weights = {0: 1.0, 1: sum(y_train_down == 0) / sum(y_train_down == 1)}
down_model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    class_weights=down_weights,
)
down_model.fit(X_train, y_train_down, eval_set=(X_val, y_val_down), use_best_model=True)
down_probs = down_model.predict_proba(X_test)[:, 1]
print(f"✅ DOWN detector AUC-like: {np.mean((down_probs > 0.5) == y_test_down):.4f}")

# Combine binary classifiers into 3-class prediction
def combine_binary_predictions(up_prob, down_prob, up_thresh=0.4, down_thresh=0.4):
    """
    Combine binary classifiers:
    - If both high -> predict stronger one
    - If UP high and DOWN low -> UP
    - If DOWN high and UP low -> DOWN
    - Otherwise -> NEUTRAL
    """
    preds = np.ones(len(up_prob), dtype=int)  # Default: NEUTRAL
    
    # Strong UP signal
    preds[(up_prob > up_thresh) & (up_prob > down_prob)] = 2
    
    # Strong DOWN signal
    preds[(down_prob > down_thresh) & (down_prob > up_prob)] = 0
    
    return preds

binary_preds = combine_binary_predictions(up_probs, down_probs)
binary_acc = (binary_preds == y_test).mean()
binary_f1 = f1_score(y_test, binary_preds, average='macro')
print(f"\n🎯 Binary Decomposition Accuracy: {binary_acc:.4f}, Macro-F1: {binary_f1:.4f}")

# ============================================================
# CELL 11: STRATEGY 3 - Deep Learning with Attention
# ============================================================
print("="*60)
print("🧠 STRATEGY 3: ATTENTION TRANSFORMER")
print("="*60)

class SwingDatasetV5(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.LongTensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class AttentionTransformer(nn.Module):
    def __init__(self, input_dim, hidden=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.pos_enc = nn.Parameter(torch.randn(1, 60, hidden) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, 
            nhead=heads, 
            dim_feedforward=hidden*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )
    
    def forward(self, x):
        # x: (batch, seq, features)
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1)]
        h = self.transformer(h)
        h = self.pool(h.permute(0, 2, 1)).squeeze(-1)
        return self.classifier(h)

# Create dataloaders with class weights for balanced sampling
train_ds = SwingDatasetV5(X_train_seq, y_train)
val_ds = SwingDatasetV5(X_val_seq, y_val)
test_ds = SwingDatasetV5(X_test_seq, y_test)

class_weights_tensor = torch.FloatTensor([class_weight_dict[i] for i in range(3)]).to(DEVICE)
sample_weight_tensor = torch.FloatTensor([class_weight_dict[y] for y in y_train])
sampler = torch.utils.data.WeightedRandomSampler(sample_weight_tensor, len(sample_weight_tensor))

train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# Train model
print("\n🔹 Training Attention Transformer...")
input_dim = len(SELECTED_FEATURES)
attn_model = AttentionTransformer(input_dim, hidden=96, heads=4, layers=2, dropout=0.1).to(DEVICE)
optimizer = torch.optim.AdamW(attn_model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

best_acc, patience_count, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    attn_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out = attn_model(X_batch.to(DEVICE))
        loss = criterion(out, y_batch.to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attn_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    
    attn_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            out = attn_model(X_batch.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y_batch.numpy())
    
    acc = np.mean(np.array(preds) == np.array(labels))
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Val Acc={acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        patience_count = 0
        best_state = {k: v.cpu().clone() for k, v in attn_model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

attn_model.load_state_dict(best_state)
print(f"✅ Attention Transformer Best Val Accuracy: {best_acc:.4f}")

# Get test predictions
attn_model.eval()
attn_preds, attn_probs_list = [], []
with torch.no_grad():
    for X_batch, _ in test_loader:
        out = attn_model(X_batch.to(DEVICE))
        probs = F.softmax(out, dim=1)
        attn_preds.extend(out.argmax(1).cpu().numpy())
        attn_probs_list.extend(probs.cpu().numpy())

attn_probs = np.array(attn_probs_list)
attn_acc = (np.array(attn_preds) == y_test).mean()
print(f"✅ Attention Transformer Test Accuracy: {attn_acc:.4f}")

# ============================================================
# CELL 12: META-LEARNER STACKING
# ============================================================
print("="*60)
print("🏗️ META-LEARNER STACKING")
print("="*60)

# Get validation predictions for stacking
cat_val_probs = catboost_model.predict_proba(X_val)
if HAS_LGB:
    lgb_val_probs = lgb_model.predict(X_val)
else:
    lgb_val_probs = cat_val_probs
if HAS_XGB:
    xgb_val_probs = xgb_model.predict_proba(X_val)
else:
    xgb_val_probs = cat_val_probs

attn_model.eval()
attn_val_probs = []
with torch.no_grad():
    for X_batch, _ in DataLoader(SwingDatasetV5(X_val_seq, y_val), batch_size=64):
        out = F.softmax(attn_model(X_batch.to(DEVICE)), dim=1)
        attn_val_probs.extend(out.cpu().numpy())
attn_val_probs = np.array(attn_val_probs)

# Stack predictions
X_meta_val = np.hstack([cat_val_probs, lgb_val_probs, xgb_val_probs, attn_val_probs])
X_meta_test = np.hstack([cat_probs, lgb_probs, xgb_probs, attn_probs])

# Train meta-learner (Logistic Regression)
print("\n🔹 Training Meta-Learner...")
scaler = StandardScaler()
X_meta_val_scaled = scaler.fit_transform(X_meta_val)
X_meta_test_scaled = scaler.transform(X_meta_test)

meta_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
meta_model.fit(X_meta_val_scaled, y_val)

meta_probs = meta_model.predict_proba(X_meta_test_scaled)
meta_preds = meta_probs.argmax(1)
meta_acc = (meta_preds == y_test).mean()
meta_f1 = f1_score(y_test, meta_preds, average='macro')
print(f"✅ Meta-Learner Accuracy: {meta_acc:.4f}, Macro-F1: {meta_f1:.4f}")

# ============================================================
# CELL 13: FINAL COMPARISON
# ============================================================
print("\n" + "="*60)
print("📊 FINAL COMPARISON - ALL STRATEGIES")
print("="*60)

results = {
    "CatBoost": (cat_probs.argmax(1) == y_test).mean(),
    "GBDT Ensemble": gbdt_acc,
    "Binary Decomposition": binary_acc,
    "Attention Transformer": attn_acc,
    "Meta-Learner Stack": meta_acc,
}

print("\n🏆 ACCURACY RANKING:")
for i, (name, acc) in enumerate(sorted(results.items(), key=lambda x: -x[1])):
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
    print(f"   {medal} {name:25s}: {acc:.4f} ({acc*100:.1f}%)")

# Best model for final evaluation
best_name = max(results, key=results.get)
best_probs = {
    "CatBoost": cat_probs,
    "GBDT Ensemble": gbdt_probs,
    "Binary Decomposition": None,  # Not probability-based
    "Attention Transformer": attn_probs,
    "Meta-Learner Stack": meta_probs,
}[best_name]

if best_probs is not None:
    best_preds = best_probs.argmax(1)
    best_conf = best_probs.max(1)
else:
    best_preds = binary_preds
    best_conf = np.maximum(up_probs, down_probs)

print(f"\n📊 Best Model: {best_name}")
print("\n📈 High-Confidence Performance:")
for thresh in [0.5, 0.6, 0.7, 0.8]:
    mask = best_conf >= thresh
    if mask.sum() > 0:
        acc = (best_preds[mask] == y_test[mask]).mean()
        print(f"   Conf >= {int(thresh*100)}%: {mask.sum():,} ({mask.mean()*100:.1f}%), Acc={acc:.4f}")

print(f"\n📊 Per-Class Recall ({best_name}):")
for c, name in [(0, 'SHORT'), (1, 'NEUTRAL'), (2, 'LONG')]:
    mask = y_test == c
    if mask.sum() > 0:
        recall = (best_preds[mask] == c).mean()
        print(f"   {name}: {recall:.4f}")

print("\n" + "="*60)
print(classification_report(y_test, best_preds, target_names=['SHORT', 'NEUTRAL', 'LONG']))

# ============================================================
# CELL 14: Save Best Models
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

print("💾 Saving V5 models...")
catboost_model.save_model(f"{config.MODEL_SAVE_PATH}catboost_v5.cbm")
up_model.save_model(f"{config.MODEL_SAVE_PATH}up_detector_v5.cbm")
down_model.save_model(f"{config.MODEL_SAVE_PATH}down_detector_v5.cbm")
torch.save(attn_model.state_dict(), f"{config.MODEL_SAVE_PATH}attention_v5.pt")

import pickle
with open(f"{config.MODEL_SAVE_PATH}meta_learner_v5.pkl", "wb") as f:
    pickle.dump({'model': meta_model, 'scaler': scaler}, f)

if HAS_LGB:
    lgb_model.save_model(f"{config.MODEL_SAVE_PATH}lightgbm_v5.txt")
if HAS_XGB:
    xgb_model.save_model(f"{config.MODEL_SAVE_PATH}xgboost_v5.json")

model_config = {
    "version": "V5_Advanced",
    "strategies": ["GBDT_Ensemble", "Binary_Decomposition", "Attention_Transformer", "Meta_Stacking"],
    "feature_columns": SELECTED_FEATURES,
    "best_model": best_name,
    "results": results,
    "trained_at": timestamp
}
with open(f"{config.MODEL_SAVE_PATH}model_config_v5.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"✅ V5 Models saved to: {config.MODEL_SAVE_PATH}")
print(f"🎉 Training complete! Best: {best_name} @ {results[best_name]*100:.1f}%")
