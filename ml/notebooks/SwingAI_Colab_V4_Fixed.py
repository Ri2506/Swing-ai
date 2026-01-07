"""
# 🚀 SwingAI Training V4 - Fixed Version
# Fixes V3 issues while keeping good changes

V3 PROBLEMS:
1. CatBoost stopped at iteration 1 (TotalF1 metric issue)
2. Focal Loss + Balanced Sampling = overcorrection
3. Feature capping too aggressive

V4 FIXES:
1. CatBoost: Use Accuracy metric, no early stopping on F1
2. Use class weights ONLY (no balanced sampling for CatBoost)
3. Less aggressive feature capping (±20% instead of ±15%)
4. Keep balanced thresholds (they helped LONG class balance)
"""

# ============================================================
# CELL 1: Install (run once, restart runtime)
# ============================================================
# %pip uninstall -y -q numpy pandas scipy scikit-learn catboost yfinance
# %pip install -q "numpy<2.0" "pandas==2.2.2" "scipy<1.13" "scikit-learn==1.5.2" \
#     "catboost==1.2.7" "yfinance>=0.2.54" "curl_cffi>=0.7.4"

# ============================================================
# CELL 2: Imports
# ============================================================
import os, json, time, random, warnings
from datetime import datetime
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings('ignore')

print(f"✅ GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   {torch.cuda.get_device_name(0)}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CELL 3: Config V4
# ============================================================
@dataclass
class ConfigV4:
    LOOKBACK_DAYS: int = 60
    PREDICTION_HORIZON: int = 5
    MIN_HISTORY_DAYS: int = 252
    
    # Keep balanced thresholds (helped class balance)
    UP_THRESHOLD: float = 0.025
    DOWN_THRESHOLD: float = -0.015
    
    TRAIN_END: str = "2024-06-30"
    VAL_END: str = "2024-09-30"
    
    # Adjusted weights based on V2 performance
    CATBOOST_WEIGHT: float = 0.35
    TFT_WEIGHT: float = 0.40
    STOCKFORMER_WEIGHT: float = 0.25
    
    BATCH_SIZE: int = 64  # Back to V2 size
    LEARNING_RATE: float = 1e-3  # Back to V2
    MAX_EPOCHS: int = 50
    PATIENCE: int = 10
    
    # CatBoost V4 - FIXED
    CATBOOST_ITERATIONS: int = 1500
    CATBOOST_DEPTH: int = 7  # Slightly deeper than V2
    CATBOOST_LR: float = 0.04
    CATBOOST_L2_REG: float = 2.0
    
    # TFT - Between V2 and V3
    TFT_HIDDEN_SIZE: int = 96
    TFT_ATTENTION_HEADS: int = 6
    TFT_DROPOUT: float = 0.15
    TFT_NUM_LAYERS: int = 2
    
    # Stockformer - Between V2 and V3
    STOCKFORMER_D_MODEL: int = 96
    STOCKFORMER_N_HEADS: int = 6
    STOCKFORMER_N_LAYERS: int = 2
    STOCKFORMER_DROPOUT: float = 0.1
    
    # Loss - REDUCED focal gamma
    FOCAL_GAMMA: float = 1.0  # Reduced from 2.0
    LABEL_SMOOTHING: float = 0.05  # Reduced from 0.1
    
    NUM_FEATURES: int = 40
    MODEL_SAVE_PATH: str = "/content/drive/MyDrive/SwingAI/models_v4/"

config = ConfigV4()

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

print(f"✅ Config V4: {len(FEATURE_NAMES)} features")

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
# CELL 5: Features V4 (Less aggressive capping)
# ============================================================
def calculate_features_v4(df):
    data = df.copy()
    
    # Price Action (same as V2)
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
    
    # SMC/ICT - V4: Use 7-day lookback (between V2's 10 and V3's 5)
    data['swing_high'] = data['High'].rolling(7, center=True).max()
    data['swing_low'] = data['Low'].rolling(7, center=True).min()
    data['prev_swing_high'] = data['swing_high'].shift(7)
    data['prev_swing_low'] = data['swing_low'].shift(7)
    data['higher_high'] = (data['swing_high'] > data['prev_swing_high']).astype(int)
    data['higher_low'] = (data['swing_low'] > data['prev_swing_low']).astype(int)
    data['lower_high'] = (data['swing_high'] < data['prev_swing_high']).astype(int)
    data['lower_low'] = (data['swing_low'] < data['prev_swing_low']).astype(int)
    
    data['structure_score'] = (data['higher_high'].rolling(5).sum() + 
                               data['higher_low'].rolling(5).sum() -
                               data['lower_high'].rolling(5).sum() - 
                               data['lower_low'].rolling(5).sum()) / 10
    
    # V4: Use 30-day range (between V2's 50 and V3's 20)
    range_high = data['High'].rolling(30).max()
    range_low = data['Low'].rolling(30).min()
    data['range_position'] = (data['Close'] - range_low) / (range_high - range_low + 1e-10)
    
    # V4: LESS AGGRESSIVE capping (±20% instead of ±15%)
    data['dist_to_swing_high'] = np.clip((data['swing_high'] - data['Close']) / data['Close'], -0.20, 0.20)
    data['dist_to_swing_low'] = np.clip((data['Close'] - data['swing_low']) / data['Close'], -0.20, 0.20)
    
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
    
    # Volume - V4: Cap at 4x (between V2's uncapped and V3's 5x)
    data['volume_ma_20'] = data['Volume'].rolling(20).mean()
    data['volume_ratio'] = np.clip(data['Volume'] / (data['volume_ma_20'] + 1e-10), 0, 4)
    data['volume_trend'] = data['Volume'].rolling(5).mean() / (data['Volume'].rolling(20).mean() + 1e-10)
    
    obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    data['obv_slope'] = obv.diff(5) / (obv.rolling(20).std() + 1e-10)
    
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['close_to_vwap'] = (data['Close'] - data['vwap']) / (data['vwap'] + 1e-10)
    
    data['buying_pressure'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    data['accumulation_score'] = (data['buying_pressure'] * data['volume_ratio']).rolling(5).mean()
    data['big_volume_day'] = (data['volume_ratio'] > 2).astype(int)
    
    # Multi-timeframe (same as V2)
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

def create_labels_v4(df):
    forward_return = df['Close'].shift(-config.PREDICTION_HORIZON) / df['Close'] - 1
    labels = pd.Series(1, index=df.index)
    labels[forward_return >= config.UP_THRESHOLD] = 2
    labels[forward_return <= config.DOWN_THRESHOLD] = 0
    return labels

print("🔧 Calculating features...")
feature_data, labels_data = {}, {}
for symbol, data in stock_data.items():
    try:
        features = calculate_features_v4(data)
        labels = create_labels_v4(data)
        if len(features) > config.LOOKBACK_DAYS + config.PREDICTION_HORIZON:
            feature_data[symbol] = features
            labels_data[symbol] = labels
    except: pass
print(f"✅ Processed {len(feature_data)} stocks")

# ============================================================
# CELL 6: Dataset (Standard - no balanced sampling for CatBoost)
# ============================================================
class SwingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {'features': self.features[idx], 'labels': self.labels[idx]}

class BalancedSwingDataset(Dataset):
    """Only for neural networks - with balanced sampling"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        class_counts = Counter(labels)
        total = len(labels)
        self.class_weights = {c: total / (3 * count) for c, count in class_counts.items()}
        self.sample_weights = torch.FloatTensor([self.class_weights[int(l)] for l in labels])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {'features': self.features[idx], 'labels': self.labels[idx]}
    def get_sampler(self): return WeightedRandomSampler(self.sample_weights, len(self.sample_weights), True)

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

# Standard dataset for CatBoost
train_dataset = SwingDataset(sequences[train_mask], labels_arr[train_mask])
val_dataset = SwingDataset(sequences[val_mask], labels_arr[val_mask])
test_dataset = SwingDataset(sequences[test_mask], labels_arr[test_mask])

# Balanced dataset for neural networks
train_balanced = BalancedSwingDataset(sequences[train_mask], labels_arr[train_mask])

print(f"📊 Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
test_dist = Counter(labels_arr[test_mask])
print(f"📊 Test Distribution: SHORT={test_dist[0]}, NEUTRAL={test_dist[1]}, LONG={test_dist[2]}")

# ============================================================
# CELL 7: Soft Focal Loss (reduced gamma)
# ============================================================
class SoftFocalLoss(nn.Module):
    """Focal loss with reduced gamma for less aggressive reweighting"""
    def __init__(self, gamma=1.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        probs = torch.exp(-ce_loss)
        focal_weight = (1 - probs) ** self.gamma
        return (focal_weight * ce_loss).mean()

criterion = SoftFocalLoss(config.FOCAL_GAMMA, config.LABEL_SMOOTHING)
print(f"✅ Soft Focal Loss (gamma={config.FOCAL_GAMMA})")

# ============================================================
# CELL 8: CatBoost V4 - FIXED!
# ============================================================
print("="*60)
print("🚀 TRAINING CATBOOST V4")
print("="*60)

X_train = train_dataset.features[:, -1, :].numpy()
y_train = train_dataset.labels.numpy()
X_val = val_dataset.features[:, -1, :].numpy()
y_val = val_dataset.labels.numpy()

# Class weights (not too extreme)
class_counts = Counter(y_train)
total = len(y_train)
# Use sqrt to reduce extreme weights
class_weights = {c: np.sqrt(total / (3 * count)) for c, count in class_counts.items()}
print(f"Class weights (sqrt): {class_weights}")

catboost_model = CatBoostClassifier(
    iterations=config.CATBOOST_ITERATIONS,
    depth=config.CATBOOST_DEPTH,
    learning_rate=config.CATBOOST_LR,
    l2_leaf_reg=config.CATBOOST_L2_REG,
    loss_function='MultiClass',
    eval_metric='Accuracy',  # FIXED: Changed from TotalF1
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100,  # FIXED: Increased from 50
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    class_weights=class_weights,
)

catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

cat_preds = catboost_model.predict(X_val)
cat_acc = (cat_preds == y_val).mean()
cat_f1 = f1_score(y_val, cat_preds, average='macro')
print(f"✅ CatBoost Val Accuracy: {cat_acc:.4f}, Macro-F1: {cat_f1:.4f}")

# Feature importance
importance = dict(zip(FEATURE_NAMES, catboost_model.feature_importances_))
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n📊 Top 10 Features:")
for i, (f, v) in enumerate(sorted_imp):
    print(f"   {i+1}. {f}: {v:.2f}")

# ============================================================
# CELL 9: TFT V4
# ============================================================
class TFTModelV4(nn.Module):
    def __init__(self):
        super().__init__()
        h = config.TFT_HIDDEN_SIZE
        heads = config.TFT_ATTENTION_HEADS
        drop = config.TFT_DROPOUT
        
        self.var_sel = nn.Sequential(
            nn.Linear(40, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 40), nn.Softmax(dim=-1)
        )
        self.lstm = nn.LSTM(40, h, config.TFT_NUM_LAYERS, batch_first=True, dropout=drop if config.TFT_NUM_LAYERS > 1 else 0)
        self.attention = nn.MultiheadAttention(h, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.grn = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h))
        self.grn_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.grn_norm = nn.LayerNorm(h)
        self.output = nn.Linear(h, 3)
    
    def forward(self, x):
        w = self.var_sel(x.mean(1))
        x = x * w.unsqueeze(1)
        o, _ = self.lstm(x)
        a, _ = self.attention(o, o, o)
        o = self.norm(o + a)
        final = o[:, -1, :]
        g = self.grn_gate(final) * self.grn(final)
        out = self.grn_norm(final + g)
        return self.output(out)

print("="*60)
print("🚀 TRAINING TFT V4")
print("="*60)

tft_model = TFTModelV4().to(DEVICE)
# Use balanced sampler for neural networks
train_loader = DataLoader(train_balanced, batch_size=config.BATCH_SIZE, sampler=train_balanced.get_sampler())
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
optimizer = torch.optim.AdamW(tft_model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

best_acc, patience_count, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    tft_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = tft_model(batch['features'].to(DEVICE))
        loss = criterion(out, batch['labels'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tft_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    
    tft_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            preds.extend(tft_model(b['features'].to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(b['labels'].numpy())
    
    acc = np.mean(np.array(preds) == np.array(labels))
    f1 = f1_score(labels, preds, average='macro')
    print(f"Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}")
    
    if acc > best_acc:
        best_acc, patience_count = acc, 0
        best_state = {k: v.cpu().clone() for k, v in tft_model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

tft_model.load_state_dict(best_state)
print(f"✅ TFT V4 Best Accuracy: {best_acc:.4f}")

# ============================================================
# CELL 10: Stockformer V4
# ============================================================
class StockformerV4(nn.Module):
    def __init__(self):
        super().__init__()
        d = config.STOCKFORMER_D_MODEL
        h = config.STOCKFORMER_N_HEADS
        n = config.STOCKFORMER_N_LAYERS
        drop = config.STOCKFORMER_DROPOUT
        
        self.trend_enc = nn.Sequential(nn.Linear(40, d), nn.LayerNorm(d), nn.GELU())
        self.seasonal_enc = nn.Sequential(nn.Linear(40, d), nn.LayerNorm(d), nn.GELU())
        self.residual_enc = nn.Sequential(nn.Linear(40, d), nn.LayerNorm(d), nn.GELU())
        
        self.pos = nn.Parameter(torch.randn(1, 60, d) * 0.02)
        
        self.trend_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, h, d*4, drop, batch_first=True), n
        )
        self.seasonal_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, h, d*4, drop, batch_first=True), n
        )
        self.residual_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, h, d*4, drop, batch_first=True), n
        )
        
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

print("="*60)
print("🚀 TRAINING STOCKFORMER V4")
print("="*60)

sf_model = StockformerV4().to(DEVICE)
optimizer = torch.optim.AdamW(sf_model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

best_acc, patience_count, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    sf_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = sf_model(batch['features'].to(DEVICE))
        loss = criterion(out, batch['labels'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    
    sf_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            preds.extend(sf_model(b['features'].to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(b['labels'].numpy())
    
    acc = np.mean(np.array(preds) == np.array(labels))
    f1 = f1_score(labels, preds, average='macro')
    print(f"Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}")
    
    if acc > best_acc:
        best_acc, patience_count = acc, 0
        best_state = {k: v.cpu().clone() for k, v in sf_model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

sf_model.load_state_dict(best_state)
print(f"✅ Stockformer V4 Best Accuracy: {best_acc:.4f}")

# ============================================================
# CELL 11: Ensemble Evaluation
# ============================================================
print("="*60)
print("📊 ENSEMBLE EVALUATION V4")
print("="*60)

X_test = test_dataset.features.numpy()
y_test = test_dataset.labels.numpy()

cat_probs = catboost_model.predict_proba(X_test[:, -1, :])
tft_model.eval()
sf_model.eval()
with torch.no_grad():
    tft_probs = F.softmax(tft_model(torch.FloatTensor(X_test).to(DEVICE)), 1).cpu().numpy()
    sf_probs = F.softmax(sf_model(torch.FloatTensor(X_test).to(DEVICE)), 1).cpu().numpy()

# Static weighted ensemble (simpler, more stable)
ensemble_probs = (config.CATBOOST_WEIGHT * cat_probs + 
                  config.TFT_WEIGHT * tft_probs + 
                  config.STOCKFORMER_WEIGHT * sf_probs)

ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_conf = np.max(ensemble_probs, axis=1)

ensemble_acc = (ensemble_preds == y_test).mean()
ensemble_f1 = f1_score(y_test, ensemble_preds, average='macro')

print(f"\n🎯 ENSEMBLE TEST ACCURACY: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
print(f"🎯 ENSEMBLE MACRO-F1: {ensemble_f1:.4f}")

print(f"\n📋 Individual Models:")
print(f"   CatBoost:    Acc={np.mean(cat_probs.argmax(1)==y_test):.4f}")
print(f"   TFT:         Acc={np.mean(tft_probs.argmax(1)==y_test):.4f}")
print(f"   Stockformer: Acc={np.mean(sf_probs.argmax(1)==y_test):.4f}")

print(f"\n📈 High-Confidence Performance:")
for thresh in [0.5, 0.6, 0.7, 0.8]:
    mask = ensemble_conf >= thresh
    if mask.sum() > 0:
        acc = (ensemble_preds[mask] == y_test[mask]).mean()
        print(f"   Conf >= {int(thresh*100)}%: {mask.sum():,} ({mask.mean()*100:.1f}%), Acc={acc:.4f}")

print(f"\n📊 Per-Class Recall:")
for c, name in [(0, 'SHORT'), (1, 'NEUTRAL'), (2, 'LONG')]:
    mask = y_test == c
    if mask.sum() > 0:
        recall = (ensemble_preds[mask] == c).mean()
        print(f"   {name}: {recall:.4f}")

# Model agreement
cat_preds = cat_probs.argmax(1)
tft_preds = tft_probs.argmax(1)
sf_preds = sf_probs.argmax(1)
agreement_3 = (cat_preds == tft_preds) & (tft_preds == sf_preds)
print(f"\n🤝 3/3 Agreement: {agreement_3.sum():,} ({agreement_3.mean()*100:.1f}%)")
if agreement_3.sum() > 0:
    print(f"   Accuracy when all agree: {(ensemble_preds[agreement_3] == y_test[agreement_3]).mean():.4f}")

print("\n" + "="*60)
print(classification_report(y_test, ensemble_preds, target_names=['SHORT', 'NEUTRAL', 'LONG']))

# ============================================================
# CELL 12: Save Models
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

print("💾 Saving V4 models...")
catboost_model.save_model(f"{config.MODEL_SAVE_PATH}catboost_v4.cbm")
torch.save(tft_model.state_dict(), f"{config.MODEL_SAVE_PATH}tft_v4.pt")
torch.save(sf_model.state_dict(), f"{config.MODEL_SAVE_PATH}stockformer_v4.pt")

model_config = {
    "version": "V4_Fixed",
    "feature_columns": FEATURE_NAMES,
    "thresholds": {"up": config.UP_THRESHOLD, "down": config.DOWN_THRESHOLD},
    "ensemble_weights": {"catboost": config.CATBOOST_WEIGHT, "tft": config.TFT_WEIGHT, "stockformer": config.STOCKFORMER_WEIGHT},
    "test_accuracy": float(ensemble_acc),
    "test_macro_f1": float(ensemble_f1),
    "trained_at": timestamp
}
with open(f"{config.MODEL_SAVE_PATH}model_config_v4.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"✅ V4 Models saved to: {config.MODEL_SAVE_PATH}")
print(f"🎉 Training complete! Accuracy: {ensemble_acc*100:.1f}%, Macro-F1: {ensemble_f1:.4f}")
