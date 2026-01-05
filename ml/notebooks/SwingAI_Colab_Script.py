"""
================================================================================
🚀 SWINGAI COLAB TRAINING SCRIPT V2
================================================================================

Copy this entire script into a Google Colab notebook cell and run it.
Make sure to enable GPU: Runtime → Change runtime type → T4 GPU

Features: 40 (Pure OHLCV-based)
Models: CatBoost + TFT + Stockformer Ensemble
Training Time: ~1.5-2 hours on T4 GPU

================================================================================
"""

# ==============================================================================
# CELL 1: INSTALL DEPENDENCIES
# ==============================================================================
# Run this first, then restart runtime if needed

"""
%pip install -q catboost==1.2.2
%pip install -q yfinance==0.2.36
%pip install -q pandas==2.1.4
%pip install -q numpy==1.26.3
%pip install -q scikit-learn==1.4.0
%pip install -q torch torchvision torchaudio
"""

# ==============================================================================
# CELL 2: IMPORTS & SETUP
# ==============================================================================

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Check GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==============================================================================
# CELL 3: CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    LOOKBACK_DAYS: int = 60
    PREDICTION_HORIZON: int = 5
    MIN_HISTORY_DAYS: int = 252
    
    UP_THRESHOLD: float = 0.03
    DOWN_THRESHOLD: float = -0.02
    
    TRAIN_END: str = "2024-06-30"
    VAL_END: str = "2024-09-30"
    
    CATBOOST_WEIGHT: float = 0.35
    TFT_WEIGHT: float = 0.35
    STOCKFORMER_WEIGHT: float = 0.30
    
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    MAX_EPOCHS: int = 50
    PATIENCE: int = 5
    
    CATBOOST_ITERATIONS: int = 1000
    CATBOOST_DEPTH: int = 6
    CATBOOST_LR: float = 0.05
    
    TFT_HIDDEN_SIZE: int = 64
    TFT_ATTENTION_HEADS: int = 4
    TFT_DROPOUT: float = 0.1
    
    STOCKFORMER_D_MODEL: int = 64
    STOCKFORMER_N_HEADS: int = 4
    STOCKFORMER_N_LAYERS: int = 2
    
    NUM_FEATURES: int = 40
    
    MODEL_SAVE_PATH: str = "/content/drive/MyDrive/SwingAI/models/"
    DATA_CACHE_PATH: str = "/content/data_cache/"

config = Config()

# 40 Features
FEATURE_NAMES = [
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

print(f"✅ Config loaded: {len(FEATURE_NAMES)} features")

# ==============================================================================
# CELL 4: STOCK UNIVERSE
# ==============================================================================

FO_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "ONGC.NS",
    "TECHM.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "COALINDIA.NS", "SBILIFE.NS", "BPCL.NS", "INDUSINDBK.NS", "TATACONSUM.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "SHRIRAMFIN.NS",
    "TRENT.NS", "POLYCAB.NS", "PERSISTENT.NS", "DIXON.NS", "TATAELXSI.NS",
    "ABB.NS", "SIEMENS.NS", "HAL.NS", "BEL.NS", "IRCTC.NS",
    "ZOMATO.NS", "COFORGE.NS", "MUTHOOTFIN.NS", "INDHOTEL.NS", "BANKBARODA.NS",
    "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "CHOLAFIN.NS", "VEDL.NS",
]

print(f"📊 Stock Universe: {len(FO_STOCKS)} F&O stocks")

# ==============================================================================
# CELL 5: DATA DOWNLOAD
# ==============================================================================

def download_stock(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if len(df) < config.MIN_HISTORY_DAYS:
            return None
        df['Symbol'] = symbol.replace('.NS', '')
        return df
    except:
        return None

def download_all_stocks(symbols, start, end):
    data = {}
    print(f"📥 Downloading {len(symbols)} stocks...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_stock, sym, start, end): sym for sym in symbols}
        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            df = future.result()
            if df is not None:
                data[symbol] = df
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{len(symbols)}")
    
    print(f"✅ Downloaded {len(data)} stocks")
    return data

# Download data
os.makedirs(config.DATA_CACHE_PATH, exist_ok=True)
stock_data = download_all_stocks(FO_STOCKS, "2019-01-01", "2024-12-31")

# ==============================================================================
# CELL 6: FEATURE ENGINEERING
# ==============================================================================

def calculate_features(df):
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
    
    features_df = data[FEATURE_NAMES].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return features_df

def create_labels(df):
    forward_return = df['Close'].shift(-config.PREDICTION_HORIZON) / df['Close'] - 1
    labels = pd.Series(1, index=df.index)
    labels[forward_return >= config.UP_THRESHOLD] = 2
    labels[forward_return <= config.DOWN_THRESHOLD] = 0
    return labels

# Calculate features
print("🔧 Calculating features...")
feature_data = {}
labels_data = {}

for symbol, data in stock_data.items():
    try:
        features = calculate_features(data)
        labels = create_labels(data)
        if len(features) > config.LOOKBACK_DAYS + config.PREDICTION_HORIZON:
            feature_data[symbol] = features
            labels_data[symbol] = labels
    except Exception as e:
        pass

print(f"✅ Processed {len(feature_data)} stocks")

# ==============================================================================
# CELL 7: PREPARE DATASETS
# ==============================================================================

class SwingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx]}

# Create sequences
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
labels = np.array(all_labels)
dates_arr = np.array(all_dates)

print(f"📊 Total sequences: {len(sequences):,}")
print(f"   Shape: {sequences.shape}")

# Split by time
train_end = pd.Timestamp(config.TRAIN_END)
val_end = pd.Timestamp(config.VAL_END)

train_mask = dates_arr <= train_end
val_mask = (dates_arr > train_end) & (dates_arr <= val_end)
test_mask = dates_arr > val_end

train_dataset = SwingDataset(sequences[train_mask], labels[train_mask])
val_dataset = SwingDataset(sequences[val_mask], labels[val_mask])
test_dataset = SwingDataset(sequences[test_mask], labels[test_mask])

print(f"   Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

# ==============================================================================
# CELL 8: TRAIN CATBOOST
# ==============================================================================

print("\n" + "="*60)
print("🚀 TRAINING CATBOOST")
print("="*60)

X_train = train_dataset.features[:, -1, :].numpy()
y_train = train_dataset.labels.numpy()
X_val = val_dataset.features[:, -1, :].numpy()
y_val = val_dataset.labels.numpy()

catboost_model = CatBoostClassifier(
    iterations=config.CATBOOST_ITERATIONS,
    depth=config.CATBOOST_DEPTH,
    learning_rate=config.CATBOOST_LR,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    class_weights={0: 1.2, 1: 0.8, 2: 1.0}
)

catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
cat_val_acc = (catboost_model.predict(X_val) == y_val).mean()
print(f"\n✅ CatBoost Val Accuracy: {cat_val_acc:.4f}")

# Feature importance
importance = dict(zip(FEATURE_NAMES, catboost_model.feature_importances_))
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\n📊 Top 10 Features:")
for i, (f, v) in enumerate(sorted_imp[:10]):
    print(f"   {i+1}. {f}: {v:.4f}")

# ==============================================================================
# CELL 9: TRAIN TFT
# ==============================================================================

print("\n" + "="*60)
print("🚀 TRAINING TFT")
print("="*60)

class TFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        h, heads, drop = config.TFT_HIDDEN_SIZE, config.TFT_ATTENTION_HEADS, config.TFT_DROPOUT
        
        self.var_sel = nn.Sequential(nn.Linear(40, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 40), nn.Softmax(dim=-1))
        self.lstm = nn.LSTM(40, h, 2, batch_first=True, dropout=drop)
        self.attention = nn.MultiheadAttention(h, heads, dropout=drop, batch_first=True)
        self.grn = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h))
        self.grn_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.grn_norm = nn.LayerNorm(h)
        self.output = nn.Linear(h, 3)
    
    def forward(self, x):
        w = self.var_sel(x.mean(1))
        x = x * w.unsqueeze(1)
        o, _ = self.lstm(x)
        a, _ = self.attention(o, o, o)
        g = self.grn_gate(a) * self.grn(a)
        out = self.grn_norm(a + g)
        return self.output(out[:, -1, :])

tft_model = TFTModel().to(DEVICE)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
optimizer = torch.optim.AdamW(tft_model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

best_acc, patience, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    tft_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = tft_model(batch['features'].to(DEVICE))
        loss = criterion(out, batch['labels'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tft_model.parameters(), 1.0)
        optimizer.step()
    
    tft_model.eval()
    correct = sum((tft_model(b['features'].to(DEVICE)).argmax(1) == b['labels'].to(DEVICE)).sum().item() 
                  for b in val_loader)
    acc = correct / len(val_dataset)
    print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
    
    if acc > best_acc:
        best_acc, patience = acc, 0
        best_state = {k: v.cpu().clone() for k, v in tft_model.state_dict().items()}
    else:
        patience += 1
        if patience >= config.PATIENCE:
            break

tft_model.load_state_dict(best_state)
print(f"✅ TFT Best Val Accuracy: {best_acc:.4f}")

# ==============================================================================
# CELL 10: TRAIN STOCKFORMER
# ==============================================================================

print("\n" + "="*60)
print("🚀 TRAINING STOCKFORMER")
print("="*60)

class StockformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        d, h, l = config.STOCKFORMER_D_MODEL, config.STOCKFORMER_N_HEADS, config.STOCKFORMER_N_LAYERS
        
        self.trend_enc = nn.Sequential(nn.Linear(40, d), nn.LayerNorm(d), nn.GELU())
        self.seasonal_enc = nn.Sequential(nn.Linear(40, d), nn.LayerNorm(d), nn.GELU())
        self.residual_enc = nn.Sequential(nn.Linear(40, d), nn.LayerNorm(d), nn.GELU())
        
        self.pos = nn.Parameter(torch.randn(1, 60, d) * 0.02)
        
        enc = nn.TransformerEncoderLayer(d, h, d*4, 0.1, batch_first=True)
        self.trend_tf = nn.TransformerEncoder(enc, l)
        self.seasonal_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, d*4, 0.1, batch_first=True), l)
        self.residual_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, d*4, 0.1, batch_first=True), l)
        
        self.fusion = nn.Sequential(nn.Linear(d*3, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.output = nn.Linear(d, 3)
    
    def forward(self, x):
        b, s, _ = x.shape
        trend = x.cumsum(1) / torch.arange(1, s+1, device=x.device).view(1, -1, 1)
        seasonal = x - trend
        
        t = self.trend_tf(self.trend_enc(trend) + self.pos[:, :s])[:, -1]
        se = self.seasonal_tf(self.seasonal_enc(seasonal) + self.pos[:, :s])[:, -1]
        r = self.residual_tf(self.residual_enc(x) + self.pos[:, :s])[:, -1]
        
        return self.output(self.fusion(torch.cat([t, se, r], -1)))

sf_model = StockformerModel().to(DEVICE)
optimizer = torch.optim.AdamW(sf_model.parameters(), lr=config.LEARNING_RATE)

best_acc, patience, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    sf_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = sf_model(batch['features'].to(DEVICE))
        loss = criterion(out, batch['labels'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)
        optimizer.step()
    
    sf_model.eval()
    correct = sum((sf_model(b['features'].to(DEVICE)).argmax(1) == b['labels'].to(DEVICE)).sum().item() 
                  for b in val_loader)
    acc = correct / len(val_dataset)
    print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
    
    if acc > best_acc:
        best_acc, patience = acc, 0
        best_state = {k: v.cpu().clone() for k, v in sf_model.state_dict().items()}
    else:
        patience += 1
        if patience >= config.PATIENCE:
            break

sf_model.load_state_dict(best_state)
print(f"✅ Stockformer Best Val Accuracy: {best_acc:.4f}")

# ==============================================================================
# CELL 11: ENSEMBLE EVALUATION
# ==============================================================================

print("\n" + "="*60)
print("📊 ENSEMBLE EVALUATION")
print("="*60)

X_test = test_dataset.features.numpy()
y_test = test_dataset.labels.numpy()

cat_probs = catboost_model.predict_proba(X_test[:, -1, :])

tft_model.eval()
with torch.no_grad():
    tft_probs = torch.softmax(tft_model(torch.FloatTensor(X_test).to(DEVICE)), 1).cpu().numpy()

sf_model.eval()
with torch.no_grad():
    sf_probs = torch.softmax(sf_model(torch.FloatTensor(X_test).to(DEVICE)), 1).cpu().numpy()

ensemble_probs = (config.CATBOOST_WEIGHT * cat_probs + 
                  config.TFT_WEIGHT * tft_probs + 
                  config.STOCKFORMER_WEIGHT * sf_probs)

ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc = (ensemble_preds == y_test).mean()

print(f"\n🎯 ENSEMBLE TEST ACCURACY: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
print(f"\nIndividual Models:")
print(f"   CatBoost:    {(np.argmax(cat_probs, 1) == y_test).mean():.4f}")
print(f"   TFT:         {(np.argmax(tft_probs, 1) == y_test).mean():.4f}")
print(f"   Stockformer: {(np.argmax(sf_probs, 1) == y_test).mean():.4f}")

print("\n" + classification_report(y_test, ensemble_preds, target_names=['SHORT', 'NEUTRAL', 'LONG']))

# ==============================================================================
# CELL 12: SAVE MODELS
# ==============================================================================

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

catboost_model.save_model(f"{config.MODEL_SAVE_PATH}/catboost_model.cbm")
torch.save(tft_model.state_dict(), f"{config.MODEL_SAVE_PATH}/tft_model.pt")
torch.save(sf_model.state_dict(), f"{config.MODEL_SAVE_PATH}/stockformer_model.pt")

config_dict = {
    'feature_names': FEATURE_NAMES,
    'num_features': 40,
    'test_accuracy': float(ensemble_acc),
    'weights': {'catboost': 0.35, 'tft': 0.35, 'stockformer': 0.30},
    'trained_at': datetime.now().isoformat()
}
with open(f"{config.MODEL_SAVE_PATH}/config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

with open(f"{config.MODEL_SAVE_PATH}/feature_importance.json", 'w') as f:
    json.dump(importance, f, indent=2)

print(f"\n✅ Models saved to: {config.MODEL_SAVE_PATH}")
print("🎉 TRAINING COMPLETE!")
