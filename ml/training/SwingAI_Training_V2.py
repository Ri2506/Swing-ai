"""
================================================================================
                    SWINGAI TRAINING PIPELINE V2
                    =============================
                    
    AI Engine: CatBoost (35%) + TFT (35%) + Stockformer (30%)
    Features: 40 (Pure OHLCV-based - No external dependencies)
    Market: Indian NSE/BSE
    
    V2 Changes:
    - Reduced from 60 to 40 features
    - Removed Market Context features (VIX, Nifty, Breadth)
    - Removed Institutional features (FII/DII)
    - These are now handled by Rule-Based Filters (separate module)
    
    Run on: Google Colab with T4/A100 GPU
    Training Time: ~1.5 hours
    
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

warnings.filterwarnings('ignore')

# Check for GPU
try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    DEVICE = 'cpu'
    print("PyTorch not available, using CPU")

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Central configuration for SwingAI V2"""
    
    # Data settings
    LOOKBACK_DAYS: int = 60          # Days of history for each prediction
    PREDICTION_HORIZON: int = 5       # Predict 5 days ahead (swing trading)
    MIN_HISTORY_DAYS: int = 252       # Minimum 1 year history required
    
    # Label thresholds (asymmetric for better risk management)
    UP_THRESHOLD: float = 0.03        # +3% = UP (LONG signal)
    DOWN_THRESHOLD: float = -0.02     # -2% = DOWN (SHORT signal)
    
    # Training settings
    TRAIN_START: str = "2019-01-01"
    TRAIN_END: str = "2024-06-30"
    VAL_END: str = "2024-09-30"
    TEST_END: str = "2024-12-31"
    
    # Model weights in ensemble
    CATBOOST_WEIGHT: float = 0.35
    TFT_WEIGHT: float = 0.35
    STOCKFORMER_WEIGHT: float = 0.30
    
    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    MAX_EPOCHS: int = 50
    PATIENCE: int = 5
    
    # CatBoost settings
    CATBOOST_ITERATIONS: int = 1000
    CATBOOST_DEPTH: int = 6
    CATBOOST_LR: float = 0.05
    
    # TFT settings
    TFT_HIDDEN_SIZE: int = 64
    TFT_ATTENTION_HEADS: int = 4
    TFT_DROPOUT: float = 0.1
    
    # Stockformer settings
    STOCKFORMER_D_MODEL: int = 64
    STOCKFORMER_N_HEADS: int = 4
    STOCKFORMER_N_LAYERS: int = 2
    STL_PERIOD: int = 5  # Weekly seasonality (5 trading days)
    
    # Feature count (V2: reduced to 40)
    NUM_FEATURES: int = 40
    
    # Paths
    MODEL_SAVE_PATH: str = "./models/"
    DATA_CACHE_PATH: str = "./data_cache/"
    
    # Feature names (40 features - pure OHLCV based)
    FEATURE_NAMES: List[str] = field(default_factory=lambda: [
        # Price Action (10)
        'return_1d', 'return_5d', 'return_10d', 'return_20d', 'volatility_20d',
        'close_to_sma_20', 'close_to_sma_50', 'rsi_14_norm', 'macd_histogram_norm', 'bb_position',
        
        # SMC/ICT (15)
        'structure_score', 'range_position', 'dist_to_swing_high', 'dist_to_swing_low',
        'in_discount', 'in_deep_discount', 'in_premium', 'near_bullish_ob', 'near_bearish_ob',
        'bullish_fvg', 'bearish_fvg', 'sweep_high', 'sweep_low', 'bos_bullish', 'bos_bearish',
        
        # Volume (8)
        'volume_ratio', 'volume_trend', 'obv_slope', 'close_to_vwap',
        'buying_pressure', 'accumulation_score', 'big_volume_day', 'higher_high',
        
        # Multi-timeframe (7)
        'daily_trend', 'weekly_trend', 'monthly_trend', 'mtf_alignment',
        'weekly_range_pos', 'monthly_range_pos', 'trend_strength',
    ])

config = Config()

# ==============================================================================
# SECTION 3: STOCK UNIVERSE
# ==============================================================================

# F&O Stocks (can short + highly liquid)
FO_STOCKS = [
    # Nifty 50 (most liquid)
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
    
    # High momentum mid-caps
    "TRENT.NS", "POLYCAB.NS", "PERSISTENT.NS", "DIXON.NS", "TATAELXSI.NS",
    "ASTRAL.NS", "COFORGE.NS", "LALPATHLAB.NS", "MUTHOOTFIN.NS", "INDHOTEL.NS",
    "ABB.NS", "SIEMENS.NS", "HAL.NS", "BEL.NS", "IRCTC.NS",
    "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "DELHIVERY.NS", "POLICYBZR.NS",
    
    # Sectoral leaders
    "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
    "VEDL.NS", "NMDC.NS", "NATIONALUM.NS", "JINDALSTEL.NS", "SAIL.NS",
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS",
    "PIIND.NS", "AARTIIND.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS", "SRF.NS",
    
    # Additional F&O stocks
    "CHOLAFIN.NS", "M&MFIN.NS", "LICHSGFIN.NS", "CANFINHOME.NS", "RECLTD.NS",
    "PFC.NS", "IRFC.NS", "HUDCO.NS", "BHEL.NS", "CUMMINSIND.NS",
    "VOLTAS.NS", "HAVELLS.NS", "CROMPTON.NS", "BLUESTARCO.NS", "AMBER.NS",
]

print(f"Stock Universe: {len(FO_STOCKS)} F&O stocks")

# ==============================================================================
# SECTION 4: DATA COLLECTION
# ==============================================================================

import yfinance as yf

class DataCollector:
    """Collects and caches stock data from Yahoo Finance"""
    
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.DATA_CACHE_PATH, exist_ok=True)
    
    def download_stock(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Download single stock data"""
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if len(df) < self.config.MIN_HISTORY_DAYS:
                print(f"  {symbol}: Insufficient data ({len(df)} days)")
                return None
            df['Symbol'] = symbol.replace('.NS', '')
            return df
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
            return None
    
    def download_all_stocks(
        self, 
        symbols: List[str], 
        start: str, 
        end: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Download all stocks in parallel with caching"""
        
        cache_file = f"{self.config.DATA_CACHE_PATH}/stock_data_{start}_{end}.pkl"
        
        # Try loading from cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        data = {}
        print(f"\nDownloading {len(symbols)} stocks...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.download_stock, sym, start, end): sym 
                for sym in symbols
            }
            
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        data[symbol] = df
                except Exception as e:
                    print(f"  {symbol}: Error - {e}")
                
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(symbols)} stocks")
        
        print(f"Successfully downloaded {len(data)} stocks")
        
        # Cache the data
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Cached data to {cache_file}")
        
        return data

# ==============================================================================
# SECTION 5: FEATURE ENGINEERING (40 FEATURES)
# ==============================================================================

class FeatureEngineerV2:
    """
    Calculates 40 features from pure OHLCV data
    
    Categories:
    - Price Action (10): Returns, volatility, RSI, MACD, BB
    - SMC/ICT (15): Market structure, zones, order blocks, FVG
    - Volume (8): Volume analysis, OBV, VWAP, accumulation
    - Multi-Timeframe (7): Daily, weekly, monthly trends
    
    Note: Market Context and Institutional features are handled
    by the Rule-Based Filter module separately.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_names = config.FEATURE_NAMES
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 40 features for a stock"""
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # ==================== PRICE ACTION FEATURES (10) ====================
        
        # Returns at different horizons
        data['return_1d'] = data['Close'].pct_change(1)
        data['return_5d'] = data['Close'].pct_change(5)
        data['return_10d'] = data['Close'].pct_change(10)
        data['return_20d'] = data['Close'].pct_change(20)
        
        # Volatility (annualized)
        data['volatility_20d'] = data['return_1d'].rolling(20).std() * np.sqrt(252)
        
        # Moving average distances
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['sma_50'] = data['Close'].rolling(50).mean()
        data['close_to_sma_20'] = (data['Close'] - data['sma_20']) / data['sma_20']
        data['close_to_sma_50'] = (data['Close'] - data['sma_50']) / data['sma_50']
        
        # RSI (14-period, normalized to -1 to +1)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['rsi_14'] = 100 - (100 / (1 + rs))
        data['rsi_14_norm'] = (data['rsi_14'] - 50) / 50  # Normalized
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram_norm'] = (data['macd'] - data['macd_signal']) / data['Close']
        
        # Bollinger Bands position (0 to 1)
        bb_mid = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        data['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # ==================== SMC/ICT FEATURES (15) ====================
        
        # Swing highs and lows (10-day rolling)
        data['swing_high'] = data['High'].rolling(10, center=True).max()
        data['swing_low'] = data['Low'].rolling(10, center=True).min()
        
        # Market structure (Higher Highs, Higher Lows)
        data['prev_swing_high'] = data['swing_high'].shift(10)
        data['prev_swing_low'] = data['swing_low'].shift(10)
        data['higher_high'] = (data['swing_high'] > data['prev_swing_high']).astype(int)
        data['higher_low'] = (data['swing_low'] > data['prev_swing_low']).astype(int)
        data['lower_high'] = (data['swing_high'] < data['prev_swing_high']).astype(int)
        data['lower_low'] = (data['swing_low'] < data['prev_swing_low']).astype(int)
        
        # Structure score (-1 to +1)
        data['structure_score'] = (
            data['higher_high'].rolling(5).sum() + 
            data['higher_low'].rolling(5).sum() -
            data['lower_high'].rolling(5).sum() - 
            data['lower_low'].rolling(5).sum()
        ) / 10
        
        # Range position (Premium/Discount zones)
        range_high = data['High'].rolling(50).max()
        range_low = data['Low'].rolling(50).min()
        data['range_position'] = (data['Close'] - range_low) / (range_high - range_low + 1e-10)
        
        # Distance to swing points
        data['dist_to_swing_high'] = (data['swing_high'] - data['Close']) / data['Close']
        data['dist_to_swing_low'] = (data['Close'] - data['swing_low']) / data['Close']
        
        # Zone indicators
        data['in_discount'] = (data['range_position'] < 0.5).astype(int)
        data['in_deep_discount'] = (data['range_position'] < 0.3).astype(int)
        data['in_premium'] = (data['range_position'] > 0.7).astype(int)
        
        # Order Blocks (simplified: big move days)
        vol_threshold = data['return_1d'].rolling(20).std() * 2
        data['big_move_up'] = (data['return_1d'] > vol_threshold).astype(int)
        data['big_move_down'] = (data['return_1d'] < -vol_threshold).astype(int)
        data['near_bullish_ob'] = data['big_move_up'].rolling(10).sum()
        data['near_bearish_ob'] = data['big_move_down'].rolling(10).sum()
        
        # Fair Value Gaps
        data['gap_up'] = ((data['Low'] > data['High'].shift(1)) & (data['return_1d'] > 0.01)).astype(int)
        data['gap_down'] = ((data['High'] < data['Low'].shift(1)) & (data['return_1d'] < -0.01)).astype(int)
        data['bullish_fvg'] = data['gap_up'].rolling(5).sum()
        data['bearish_fvg'] = data['gap_down'].rolling(5).sum()
        
        # Liquidity sweeps
        data['sweep_high'] = (
            (data['High'] > data['swing_high'].shift(1)) & 
            (data['Close'] < data['swing_high'].shift(1))
        ).astype(int)
        data['sweep_low'] = (
            (data['Low'] < data['swing_low'].shift(1)) & 
            (data['Close'] > data['swing_low'].shift(1))
        ).astype(int)
        
        # Break of Structure
        data['bos_bullish'] = (
            (data['Close'] > data['swing_high'].shift(1)) & 
            (data['higher_high'] == 1)
        ).astype(int)
        data['bos_bearish'] = (
            (data['Close'] < data['swing_low'].shift(1)) & 
            (data['lower_low'] == 1)
        ).astype(int)
        
        # ==================== VOLUME FEATURES (8) ====================
        
        # Volume ratios
        data['volume_ma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / (data['volume_ma_20'] + 1e-10)
        data['volume_trend'] = data['Volume'].rolling(5).mean() / (data['Volume'].rolling(20).mean() + 1e-10)
        
        # On-Balance Volume slope
        obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
        data['obv_slope'] = obv.diff(5) / (obv.rolling(20).std() + 1e-10)
        
        # VWAP distance
        data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data['close_to_vwap'] = (data['Close'] - data['vwap']) / (data['vwap'] + 1e-10)
        
        # Buying pressure (where did price close in the range)
        data['buying_pressure'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
        
        # Accumulation score
        data['accumulation_score'] = (data['buying_pressure'] * data['volume_ratio']).rolling(5).mean()
        
        # Big volume day flag
        data['big_volume_day'] = (data['volume_ratio'] > 2).astype(int)
        
        # ==================== MULTI-TIMEFRAME FEATURES (7) ====================
        
        # Daily trend
        data['daily_trend'] = (data['Close'] > data['sma_20']).astype(int)
        
        # Weekly trend (5 trading days)
        data['weekly_close'] = data['Close'].rolling(5).mean()
        data['weekly_high'] = data['High'].rolling(5).max()
        data['weekly_low'] = data['Low'].rolling(5).min()
        data['weekly_trend'] = (data['weekly_close'] > data['weekly_close'].shift(5)).astype(int)
        
        # Monthly trend (21 trading days)
        data['monthly_close'] = data['Close'].rolling(21).mean()
        data['monthly_trend'] = (data['monthly_close'] > data['monthly_close'].shift(21)).astype(int)
        
        # MTF alignment (0 to 1, 1 = all bullish)
        data['mtf_alignment'] = (data['daily_trend'] + data['weekly_trend'] + data['monthly_trend']) / 3
        
        # Range positions
        data['weekly_range_pos'] = (data['Close'] - data['weekly_low']) / (data['weekly_high'] - data['weekly_low'] + 1e-10)
        data['monthly_range_pos'] = (
            data['Close'] - data['Low'].rolling(21).min()
        ) / (
            data['High'].rolling(21).max() - data['Low'].rolling(21).min() + 1e-10
        )
        
        # Trend strength
        data['trend_strength'] = abs(data['close_to_sma_20']) + abs(data['close_to_sma_50'])
        
        # ==================== SELECT FINAL 40 FEATURES ====================
        
        features_df = data[self.feature_names].copy()
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Clip extreme values
        for col in features_df.columns:
            p1, p99 = features_df[col].quantile([0.01, 0.99])
            features_df[col] = features_df[col].clip(p1, p99)
        
        return features_df
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels for swing trading
        
        Labels:
        - 0: DOWN (price drops >= 2% in next 5 days) → SHORT
        - 1: SIDEWAYS (price stays within -2% to +3%) → NO TRADE
        - 2: UP (price rises >= 3% in next 5 days) → LONG
        """
        
        # Forward return over prediction horizon
        forward_return = df['Close'].shift(-self.config.PREDICTION_HORIZON) / df['Close'] - 1
        
        # Create labels
        labels = pd.Series(1, index=df.index)  # Default: SIDEWAYS
        labels[forward_return >= self.config.UP_THRESHOLD] = 2    # UP
        labels[forward_return <= self.config.DOWN_THRESHOLD] = 0  # DOWN
        
        return labels

# ==============================================================================
# SECTION 6: DATASET PREPARATION
# ==============================================================================

import torch
from torch.utils.data import Dataset, DataLoader

class SwingDataset(Dataset):
    """PyTorch Dataset for SwingAI"""
    
    def __init__(
        self, 
        features: np.ndarray,  # Shape: (samples, lookback, num_features)
        labels: np.ndarray,    # Shape: (samples,)
        symbols: List[str] = None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.symbols = symbols
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }


def prepare_sequences(
    stock_data: Dict[str, pd.DataFrame],
    feature_data: Dict[str, pd.DataFrame],
    labels_data: Dict[str, pd.Series],
    config: Config
) -> Tuple[np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Prepare sequences for training
    
    Returns:
    - sequences: (N, lookback, features)
    - labels: (N,)
    - symbols: List of stock symbols
    - dates: List of dates for each sample
    """
    
    all_sequences = []
    all_labels = []
    all_symbols = []
    all_dates = []
    
    lookback = config.LOOKBACK_DAYS
    horizon = config.PREDICTION_HORIZON
    
    for symbol in feature_data.keys():
        features = feature_data[symbol].values
        labels = labels_data[symbol].values
        dates = feature_data[symbol].index
        
        # Create sequences
        for i in range(lookback, len(features) - horizon):
            seq = features[i-lookback:i]
            label = labels[i]
            
            # Skip if any NaN
            if not np.isnan(label) and not np.any(np.isnan(seq)):
                all_sequences.append(seq)
                all_labels.append(int(label))
                all_symbols.append(symbol)
                all_dates.append(dates[i])
    
    return (
        np.array(all_sequences), 
        np.array(all_labels), 
        all_symbols,
        all_dates
    )


def create_data_splits(
    sequences: np.ndarray,
    labels: np.ndarray,
    symbols: List[str],
    dates: List[pd.Timestamp],
    config: Config
) -> Dict[str, SwingDataset]:
    """
    Create train/val/test splits based on time
    
    Training: 2019-01-01 to 2024-06-30
    Validation: 2024-07-01 to 2024-09-30
    Test: 2024-10-01 to 2024-12-31
    """
    
    # Convert dates to numpy array for filtering
    dates_arr = np.array(dates)
    
    train_end = pd.Timestamp(config.TRAIN_END)
    val_end = pd.Timestamp(config.VAL_END)
    
    # Create masks
    train_mask = dates_arr <= train_end
    val_mask = (dates_arr > train_end) & (dates_arr <= val_end)
    test_mask = dates_arr > val_end
    
    datasets = {
        'train': SwingDataset(
            sequences[train_mask], 
            labels[train_mask],
            [s for s, m in zip(symbols, train_mask) if m]
        ),
        'val': SwingDataset(
            sequences[val_mask], 
            labels[val_mask],
            [s for s, m in zip(symbols, val_mask) if m]
        ),
        'test': SwingDataset(
            sequences[test_mask], 
            labels[test_mask],
            [s for s, m in zip(symbols, test_mask) if m]
        )
    }
    
    print(f"\nDataset Splits:")
    print(f"  Train: {len(datasets['train']):,} samples (until {config.TRAIN_END})")
    print(f"  Val:   {len(datasets['val']):,} samples (until {config.VAL_END})")
    print(f"  Test:  {len(datasets['test']):,} samples (until {config.TEST_END})")
    
    # Label distribution
    for split_name, dataset in datasets.items():
        labels_np = dataset.labels.numpy()
        down = (labels_np == 0).sum()
        side = (labels_np == 1).sum()
        up = (labels_np == 2).sum()
        print(f"  {split_name} labels: DOWN={down}, SIDE={side}, UP={up}")
    
    return datasets

# ==============================================================================
# SECTION 7: MODEL 1 - CATBOOST
# ==============================================================================

from catboost import CatBoostClassifier

class CatBoostModel:
    """
    CatBoost Gradient Boosting Model
    
    Best for: Tabular data, handles categorical features well
    Weight in ensemble: 35%
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.name = "CatBoost"
        self.feature_importance = None
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset) -> Dict:
        """Train CatBoost model"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING CATBOOST MODEL")
        print(f"{'='*60}")
        
        # Flatten sequences - use last day features only for CatBoost
        X_train = train_dataset.features[:, -1, :].numpy()
        y_train = train_dataset.labels.numpy()
        
        X_val = val_dataset.features[:, -1, :].numpy()
        y_val = val_dataset.labels.numpy()
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {X_train.shape[1]}")
        
        # Initialize model
        self.model = CatBoostClassifier(
            iterations=self.config.CATBOOST_ITERATIONS,
            depth=self.config.CATBOOST_DEPTH,
            learning_rate=self.config.CATBOOST_LR,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            task_type='GPU' if torch.cuda.is_available() else 'CPU',
            # Class weights for imbalanced data
            class_weights={0: 1.2, 1: 0.8, 2: 1.0}
        )
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Get feature importance
        self.feature_importance = dict(zip(
            self.config.FEATURE_NAMES,
            self.model.feature_importances_
        ))
        
        # Evaluate
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        train_acc = (train_preds == y_train).mean()
        val_acc = (val_preds == y_val).mean()
        
        print(f"\nCatBoost Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        # Print top 10 features
        print(f"\nTop 10 Important Features:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_features[:10]):
            print(f"  {i+1}. {feat}: {imp:.4f}")
        
        return {
            'train_acc': train_acc, 
            'val_acc': val_acc,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        if features.ndim == 3:
            features = features[:, -1, :]  # Use last day
        
        probs = self.model.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_model(f"{path}/catboost_model.cbm")
        
        # Save feature importance
        with open(f"{path}/catboost_feature_importance.json", 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
    
    def load(self, path: str):
        """Load model"""
        self.model = CatBoostClassifier()
        self.model.load_model(f"{path}/catboost_model.cbm")
        
        if os.path.exists(f"{path}/catboost_feature_importance.json"):
            with open(f"{path}/catboost_feature_importance.json", 'r') as f:
                self.feature_importance = json.load(f)

# ==============================================================================
# SECTION 8: MODEL 2 - TFT (Temporal Fusion Transformer)
# ==============================================================================

import torch.nn as nn

class TFTModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer
    
    Architecture:
    - Variable Selection Network (learns feature importance)
    - LSTM Encoder (captures temporal patterns)
    - Multi-Head Attention (learns long-range dependencies)
    - Gated Residual Network (controls information flow)
    
    Weight in ensemble: 35%
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        num_features = config.NUM_FEATURES
        hidden_size = config.TFT_HIDDEN_SIZE
        num_heads = config.TFT_ATTENTION_HEADS
        dropout = config.TFT_DROPOUT
        
        # Variable Selection Network
        self.variable_selection = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_features),
            nn.Softmax(dim=-1)
        )
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated Residual Network
        self.grn_fc1 = nn.Linear(hidden_size, hidden_size)
        self.grn_fc2 = nn.Linear(hidden_size, hidden_size)
        self.grn_gate = nn.Linear(hidden_size, hidden_size)
        self.grn_norm = nn.LayerNorm(hidden_size)
        self.grn_dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output = nn.Linear(hidden_size, 3)  # 3 classes
    
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        returns: (batch, 3) logits
        """
        batch_size, seq_len, num_features = x.shape
        
        # Variable selection
        var_weights = self.variable_selection(x.mean(dim=1))
        x = x * var_weights.unsqueeze(1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated Residual Network
        grn_out = torch.relu(self.grn_fc1(attn_out))
        grn_out = self.grn_dropout(self.grn_fc2(grn_out))
        gate = torch.sigmoid(self.grn_gate(attn_out))
        grn_out = self.grn_norm(attn_out + gate * grn_out)
        
        # Take last time step
        final = grn_out[:, -1, :]
        
        return self.output(final)


class TFTTrainer:
    """Trainer for TFT model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TFTModel(config).to(self.device)
        self.name = "TFT"
        self.best_state = None
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset) -> Dict:
        """Train TFT model"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING TFT MODEL")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        history = {'train_acc': [], 'val_acc': []}
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(features)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            scheduler.step(val_acc)
            
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:2d}/{self.config.MAX_EPOCHS} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        
        print(f"\nTFT Results:")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        return {'train_acc': train_acc, 'val_acc': best_val_acc, 'history': history}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/tft_model.pt")
    
    def load(self, path: str):
        """Load model"""
        self.model.load_state_dict(
            torch.load(f"{path}/tft_model.pt", map_location=self.device)
        )

# ==============================================================================
# SECTION 9: MODEL 3 - STOCKFORMER
# ==============================================================================

class StockformerModel(nn.Module):
    """
    Stockformer: Transformer with STL-inspired decomposition
    
    Architecture:
    - Trend Encoder (captures long-term direction)
    - Seasonal Encoder (captures recurring patterns)
    - Residual Encoder (captures noise/anomalies)
    - Cross-Attention Fusion
    
    Weight in ensemble: 30%
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        num_features = config.NUM_FEATURES
        d_model = config.STOCKFORMER_D_MODEL
        n_heads = config.STOCKFORMER_N_HEADS
        n_layers = config.STOCKFORMER_N_LAYERS
        dropout = 0.1
        
        # Trend encoder (moving average approximation)
        self.trend_encoder = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Seasonal encoder (deviations from trend)
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual encoder (raw features)
        self.residual_encoder = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.LOOKBACK_DAYS, d_model) * 0.02
        )
        
        # Transformer encoders for each component
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.trend_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.seasonal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True, activation='gelu'),
            num_layers=n_layers
        )
        self.residual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True, activation='gelu'),
            num_layers=n_layers
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Output
        self.output = nn.Linear(d_model, 3)
    
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        batch_size, seq_len, _ = x.shape
        
        # Approximate STL decomposition
        # Trend: cumulative mean
        trend = x.cumsum(dim=1) / torch.arange(1, seq_len+1, device=x.device).view(1, -1, 1)
        # Seasonal: deviation from trend
        seasonal = x - trend
        # Residual: original (will learn to extract residual patterns)
        residual = x
        
        # Encode each component
        trend_enc = self.trend_encoder(trend) + self.pos_encoding[:, :seq_len, :]
        seasonal_enc = self.seasonal_encoder(seasonal) + self.pos_encoding[:, :seq_len, :]
        residual_enc = self.residual_encoder(residual) + self.pos_encoding[:, :seq_len, :]
        
        # Transform
        trend_out = self.trend_transformer(trend_enc)
        seasonal_out = self.seasonal_transformer(seasonal_enc)
        residual_out = self.residual_transformer(residual_enc)
        
        # Take last time step
        trend_final = trend_out[:, -1, :]
        seasonal_final = seasonal_out[:, -1, :]
        residual_final = residual_out[:, -1, :]
        
        # Fuse
        combined = torch.cat([trend_final, seasonal_final, residual_final], dim=-1)
        fused = self.fusion(combined)
        
        return self.output(fused)


class StockformerTrainer:
    """Trainer for Stockformer model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StockformerModel(config).to(self.device)
        self.name = "Stockformer"
        self.best_state = None
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset) -> Dict:
        """Train Stockformer model"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING STOCKFORMER MODEL")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.MAX_EPOCHS
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Training
            self.model.train()
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = train_correct / train_total
            scheduler.step()
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(features)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1:2d}/{self.config.MAX_EPOCHS} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        
        print(f"\nStockformer Results:")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        return {'train_acc': train_acc, 'val_acc': best_val_acc}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/stockformer_model.pt")
    
    def load(self, path: str):
        """Load model"""
        self.model.load_state_dict(
            torch.load(f"{path}/stockformer_model.pt", map_location=self.device)
        )

# ==============================================================================
# SECTION 10: ENSEMBLE MODEL
# ==============================================================================

class SwingAIEnsemble:
    """
    Ensemble of CatBoost + TFT + Stockformer
    Weights: 35% + 35% + 30%
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize models
        self.catboost = CatBoostModel(config)
        self.tft = TFTTrainer(config)
        self.stockformer = StockformerTrainer(config)
        
        self.models = {
            'catboost': self.catboost,
            'tft': self.tft,
            'stockformer': self.stockformer
        }
        
        self.weights = {
            'catboost': config.CATBOOST_WEIGHT,
            'tft': config.TFT_WEIGHT,
            'stockformer': config.STOCKFORMER_WEIGHT
        }
        
        self.trained = False
        self.feature_importance = None
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset) -> Dict:
        """Train all models in ensemble"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING SWINGAI ENSEMBLE (V2 - 40 Features)")
        print(f"{'='*60}")
        print(f"Weights: CatBoost={self.weights['catboost']*100}%, "
              f"TFT={self.weights['tft']*100}%, "
              f"Stockformer={self.weights['stockformer']*100}%")
        
        results = {}
        
        # Train each model
        results['catboost'] = self.catboost.train(train_dataset, val_dataset)
        results['tft'] = self.tft.train(train_dataset, val_dataset)
        results['stockformer'] = self.stockformer.train(train_dataset, val_dataset)
        
        # Store feature importance from CatBoost
        self.feature_importance = results['catboost'].get('feature_importance')
        
        self.trained = True
        
        # Summary
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"\nValidation Accuracies:")
        for name, res in results.items():
            print(f"  {name}: {res['val_acc']:.4f}")
        
        # Weighted ensemble accuracy (approximate)
        weighted_acc = sum(
            results[name]['val_acc'] * self.weights[name] 
            for name in results
        )
        print(f"\nWeighted Ensemble (estimated): {weighted_acc:.4f}")
        
        return results
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Ensemble prediction with weighted average
        
        Args:
            features: (N, seq_len, num_features) array
            
        Returns:
            Dict with predictions, confidence, individual model outputs
        """
        
        if not self.trained:
            raise ValueError("Models not trained yet!")
        
        # Get predictions from each model
        cat_preds, cat_probs = self.catboost.predict(features)
        tft_preds, tft_probs = self.tft.predict(features)
        sf_preds, sf_probs = self.stockformer.predict(features)
        
        # Weighted average of probabilities
        ensemble_probs = (
            self.weights['catboost'] * cat_probs +
            self.weights['tft'] * tft_probs +
            self.weights['stockformer'] * sf_probs
        )
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)
        
        # Model agreement
        all_preds = np.stack([cat_preds, tft_preds, sf_preds], axis=1)
        agreements = (all_preds == ensemble_preds[:, None]).sum(axis=1)
        
        # Boost confidence when all models agree
        agreement_boost = np.where(agreements == 3, 0.1, 0)
        confidences = np.minimum(confidences + agreement_boost, 1.0)
        
        return {
            'direction': ensemble_preds,
            'confidence': confidences * 100,
            'probabilities': ensemble_probs,
            'individual': {
                'catboost': {'pred': cat_preds, 'prob': cat_probs},
                'tft': {'pred': tft_preds, 'prob': tft_probs},
                'stockformer': {'pred': sf_preds, 'prob': sf_probs}
            },
            'agreement': agreements
        }
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict for single sample (convenience method)"""
        if features.ndim == 2:
            features = features[np.newaxis, :]
        
        result = self.predict(features)
        
        direction_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}
        
        return {
            'direction': direction_map[result['direction'][0]],
            'confidence': round(result['confidence'][0], 2),
            'probabilities': {
                'SHORT': round(result['probabilities'][0, 0] * 100, 2),
                'NEUTRAL': round(result['probabilities'][0, 1] * 100, 2),
                'LONG': round(result['probabilities'][0, 2] * 100, 2)
            },
            'model_agreement': f"{result['agreement'][0]}/3",
            'individual_scores': {
                'catboost': round(result['individual']['catboost']['prob'][0, result['direction'][0]] * 100, 2),
                'tft': round(result['individual']['tft']['prob'][0, result['direction'][0]] * 100, 2),
                'stockformer': round(result['individual']['stockformer']['prob'][0, result['direction'][0]] * 100, 2)
            }
        }
    
    def save(self, path: str):
        """Save all models"""
        os.makedirs(path, exist_ok=True)
        
        self.catboost.save(path)
        self.tft.save(path)
        self.stockformer.save(path)
        
        # Save config
        config_dict = {
            'weights': self.weights,
            'num_features': self.config.NUM_FEATURES,
            'feature_names': self.config.FEATURE_NAMES,
            'lookback_days': self.config.LOOKBACK_DAYS,
            'prediction_horizon': self.config.PREDICTION_HORIZON
        }
        with open(f"{path}/ensemble_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Models saved to {path}")
    
    def load(self, path: str):
        """Load all models"""
        self.catboost.load(path)
        self.tft.load(path)
        self.stockformer.load(path)
        
        with open(f"{path}/ensemble_config.json", 'r') as f:
            config_dict = json.load(f)
            self.weights = config_dict['weights']
        
        self.trained = True
        print(f"Models loaded from {path}")

# ==============================================================================
# SECTION 11: EVALUATION
# ==============================================================================

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_ensemble(ensemble: SwingAIEnsemble, test_dataset: SwingDataset) -> Dict:
    """Comprehensive evaluation of ensemble"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING ENSEMBLE ON TEST SET")
    print(f"{'='*60}")
    
    features = test_dataset.features.numpy()
    labels = test_dataset.labels.numpy()
    
    # Get predictions
    results = ensemble.predict(features)
    
    # Overall accuracy
    accuracy = (results['direction'] == labels).mean()
    
    print(f"\n📊 Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(
        labels, 
        results['direction'],
        target_names=['SHORT', 'NEUTRAL', 'LONG']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(labels, results['direction'])
    print(f"📊 Confusion Matrix:")
    print(f"              Predicted")
    print(f"            SHORT  NEUT  LONG")
    print(f"Actual SHORT {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
    print(f"       NEUT  {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
    print(f"       LONG  {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")
    
    # Agreement analysis
    print(f"\n🤝 Model Agreement Analysis:")
    for i in [1, 2, 3]:
        mask = results['agreement'] == i
        if mask.sum() > 0:
            acc = (results['direction'][mask] == labels[mask]).mean()
            print(f"  {i}/3 models agree: {mask.sum():,} samples, accuracy: {acc:.4f}")
    
    # Confidence analysis
    print(f"\n📈 Confidence Analysis:")
    for threshold in [50, 60, 70, 80, 90]:
        mask = results['confidence'] >= threshold
        if mask.sum() > 0:
            acc = (results['direction'][mask] == labels[mask]).mean()
            coverage = mask.mean() * 100
            print(f"  Confidence >= {threshold}%: {mask.sum():,} samples ({coverage:.1f}%), accuracy: {acc:.4f}")
    
    # High confidence + full agreement (best signals)
    high_conf_agree = (results['confidence'] >= 70) & (results['agreement'] == 3)
    if high_conf_agree.sum() > 0:
        acc = (results['direction'][high_conf_agree] == labels[high_conf_agree]).mean()
        print(f"\n⭐ Best Signals (conf>=70% + 3/3 agree): {high_conf_agree.sum():,} samples, accuracy: {acc:.4f}")
    
    # Per-model accuracy
    print(f"\n🔬 Individual Model Test Accuracy:")
    for name in ['catboost', 'tft', 'stockformer']:
        acc = (results['individual'][name]['pred'] == labels).mean()
        print(f"  {name}: {acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'results': results
    }

# ==============================================================================
# SECTION 12: MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("        SWINGAI TRAINING PIPELINE V2")
    print("        40 Features | Pure OHLCV Based")
    print("        CatBoost + TFT + Stockformer")
    print("="*60)
    print(f"\nStart time: {datetime.now()}")
    print(f"Device: {DEVICE}")
    
    # Initialize
    config = Config()
    collector = DataCollector(config)
    engineer = FeatureEngineerV2(config)
    
    # Step 1: Download data
    print(f"\n{'='*60}")
    print(f"STEP 1: DOWNLOADING DATA")
    print(f"{'='*60}")
    
    start_date = "2018-01-01"  # Extra year for feature calculation
    end_date = "2024-12-31"
    
    stock_data = collector.download_all_stocks(
        FO_STOCKS[:80],  # Use 80 stocks for training
        start_date, 
        end_date,
        use_cache=True
    )
    
    # Step 2: Calculate features
    print(f"\n{'='*60}")
    print(f"STEP 2: CALCULATING FEATURES")
    print(f"{'='*60}")
    
    feature_data = {}
    labels_data = {}
    
    for symbol, data in stock_data.items():
        try:
            features = engineer.calculate_features(data)
            labels = engineer.create_labels(data)
            
            if len(features) > config.LOOKBACK_DAYS + config.PREDICTION_HORIZON:
                feature_data[symbol] = features
                labels_data[symbol] = labels
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    print(f"Processed {len(feature_data)} stocks")
    print(f"Features: {len(engineer.feature_names)}")
    
    # Step 3: Prepare sequences
    print(f"\n{'='*60}")
    print(f"STEP 3: PREPARING SEQUENCES")
    print(f"{'='*60}")
    
    sequences, labels, symbols, dates = prepare_sequences(
        stock_data, feature_data, labels_data, config
    )
    
    print(f"Total sequences: {len(sequences):,}")
    print(f"Sequence shape: {sequences.shape}")
    
    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution:")
    for u, c in zip(unique, counts):
        label_name = ['SHORT', 'NEUTRAL', 'LONG'][u]
        print(f"  {label_name}: {c:,} ({c/len(labels)*100:.1f}%)")
    
    # Create datasets
    datasets = create_data_splits(sequences, labels, symbols, dates, config)
    
    # Step 4: Train ensemble
    print(f"\n{'='*60}")
    print(f"STEP 4: TRAINING ENSEMBLE")
    print(f"{'='*60}")
    
    ensemble = SwingAIEnsemble(config)
    training_results = ensemble.train(datasets['train'], datasets['val'])
    
    # Step 5: Evaluate
    print(f"\n{'='*60}")
    print(f"STEP 5: EVALUATION")
    print(f"{'='*60}")
    
    eval_results = evaluate_ensemble(ensemble, datasets['test'])
    
    # Step 6: Save models
    print(f"\n{'='*60}")
    print(f"STEP 6: SAVING MODELS")
    print(f"{'='*60}")
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    ensemble.save(config.MODEL_SAVE_PATH)
    
    # Print feature importance
    if ensemble.feature_importance:
        print(f"\n📊 Top 15 Feature Importance (CatBoost):")
        sorted_features = sorted(
            ensemble.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for i, (feat, imp) in enumerate(sorted_features[:15]):
            print(f"  {i+1:2d}. {feat:25s}: {imp:.4f}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"End time: {datetime.now()}")
    print(f"Models saved to: {config.MODEL_SAVE_PATH}")
    
    return ensemble, eval_results


if __name__ == "__main__":
    ensemble, results = main()
