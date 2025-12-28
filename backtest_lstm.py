import asyncio
import pandas as pd
import numpy as np
import logging
import os
import time
from dotenv import load_dotenv

# Load env variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

from iqclient import IQOptionAPI
# Import LSTM predictor
try:
    from ml_lstm import predict_lstm, get_model_and_scaler, prepare_sequences, prepare_features
    # We might need to expose the model directly or use the function. 
    # The function 'predict_lstm' loads the model fresh every time? No, it uses global cache.
except ImportError:
    print("ml_lstm.py not found or TF missing.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Re-implement fetching to ensure we get features
async def fetch_and_prepare_multi_asset(api, target_asset, correlated_assets, timeframe, count):
    """
    Fetches historical candles for Target + Correlated assets and merges them.
    Returns a 'Wide' DataFrame with features.
    """
    all_assets = [target_asset] + correlated_assets
    asset_dfs = {}
    
    logger.info(f"Fetching {count} candles for {len(all_assets)} assets...")
    
    for asset in all_assets:
        # Fetch data
        candles = api.get_candle_history(asset, count, timeframe)
        if not candles:
            logger.warning(f"No candles for {asset}")
            return None
            
        df = pd.DataFrame(candles)
        
        # Ensure numeric
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
                
        if 'from' in df.columns:
            df['time'] = pd.to_datetime(df['from'], unit='s')
            
        # Standardize columns for correlated assets
        if asset != target_asset:
             # Keep only useful columns for correlation
            cols_to_keep = ['time', 'open', 'close', 'min', 'max', 'volume']
            df = df[cols_to_keep]
            
            # Rename columns: e.g. open -> USDJPY_open
            rename_map = {c: f"{asset}_{c}" for c in df.columns if c != 'time'}
            df = df.rename(columns=rename_map)
            
        asset_dfs[asset] = df
        
    # Merge
    final_df = asset_dfs[target_asset]
    for asset in correlated_assets:
        if asset in asset_dfs:
            final_df = pd.merge(final_df, asset_dfs[asset], on='time', how='inner')
            
    # Calculate Features (on the merged DF, primarily for Target columns)
    from ml_utils import prepare_features
    final_df = prepare_features(final_df)
    final_df = final_df.reset_index(drop=True)
    
    logger.info(f"Final Data Shape for {target_asset}: {final_df.shape}")
    return final_df


# Import XGBoost utils
try:
    from ml_utils import load_model as load_xgb_model
    import joblib
except ImportError:
    print("ml_utils not found.")


def apply_hybrid_strategy(df):
    """
    Applies Hybrid Prediction (LSTM + XGBoost).
    """
    df['signal'] = 0
    df['prob'] = 0.5
    df['lstm_prob'] = 0.5
    df['xgb_prob'] = 0.5
    
    # 1. Load LSTM
    from ml_lstm import get_model_and_scaler
    lstm_model, scaler = get_model_and_scaler()
    
    # 2. Load XGBoost
    xgb_model = load_xgb_model()
    
    if not lstm_model or not xgb_model:
        logger.error("One or both models failed to load.")
        return df
        
    # --- LSTM BATCH PREDICTION (Vectorized) ---
    SEQ_LENGTH = 10
    drop_raw = ['open', 'close', 'min', 'max', 'volume', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']
    drop_meta = ['time', 'outcome', 'signal', 'asset', 'from', 'to', 'next_close', 'next_open', 'id', 'at']
    
    # Prepare features for LSTM
    features_df = df.drop(columns=[c for c in list(df.columns) if c in drop_raw + drop_meta], errors='ignore')
    
    # Robust Feature Alignment
    if hasattr(scaler, "feature_names_in_"):
        # Filter to keep ONLY columns expected by scaler
        # If columns are missing, this will fail (good).
        # If extra columns exist, they are dropped (fix).
        try:
            features_df = features_df[scaler.feature_names_in_]
        except KeyError as e:
            logger.error(f"Missing features for Scaler: {e}")
            return df
            
    feature_values = features_df.values
    
    try:
        scaled_features = scaler.transform(feature_values)
    except Exception as e:
        logger.error(f"Scaler transformation failed: {e}")
        return df

    sequences = []
    valid_indices = []
    
    for i in range(SEQ_LENGTH, len(df)):
        seq = scaled_features[i-SEQ_LENGTH : i]
        sequences.append(seq)
        valid_indices.append(i)
        
    if not sequences: return df
    
    X_lstm = np.array(sequences)
    lstm_probs = lstm_model.predict(X_lstm, verbose=0)
    
    # --- XGBOOST BATCH PREDICTION ---
    # XGBoost was trained on the SAME features (minus raw/meta).
    # We must match the columns exactly.
    if hasattr(xgb_model, "feature_names_in_"):
        # Select columns that match model
        xgb_input = features_df.copy()
        
        # Add missing as 0
        missing = set(xgb_model.feature_names_in_) - set(xgb_input.columns)
        for c in missing: xgb_input[c] = 0
            
        xgb_input = xgb_input[xgb_model.feature_names_in_]
        
        # Predict Probabilities
        xgb_probs = xgb_model.predict_proba(xgb_input) # [ [prob_0, prob_1], ... ]
        # Align XGB probs to valid_indices? XGB predicts on ROW `i`. 
        # LSTM predicts on sequence ending at `i`.
        # So for row `i`, we use `xgb_probs[i]`.
        
    else:
        logger.warning("XGBoost model has no feature names. Assuming column order match.")
        xgb_probs = xgb_model.predict_proba(features_df)

    # --- COMBINE PREDICTIONS ---
    for idx, list_idx in enumerate(valid_indices):
        # 1. LSTM Prob
        l_prob = lstm_probs[idx][0]
        
        # 2. XGB Prob (for the same row `list_idx`)
        x_prob = xgb_probs[list_idx][1] # Probability of Class 1 (Win/Up?)
        # Wait, Class 1 in `ml_utils` logic was "Next Candle Green" or "Win"?
        # In `collect_training_data`, `outcome` was (next_close > next_open). 
        # So Class 1 = Call/Green.
        
        # Store
        df.at[list_idx, 'lstm_prob'] = l_prob
        df.at[list_idx, 'xgb_prob'] = x_prob
        
        # 3. Hybrid Logic
        # Weighted Average: Maybe give more weight to XGB if it had 56% acc?
        combined_prob = (l_prob * 0.4) + (x_prob * 0.6)
        df.at[list_idx, 'prob'] = combined_prob
        
        # Strategy Thresholds (Conservative)
        if combined_prob > 0.60: 
            df.at[list_idx, 'signal'] = 1 # CALL
        elif combined_prob < 0.40:
            df.at[list_idx, 'signal'] = -1 # PUT
            
        # Consensus Strategy (Alternative)
        # if l_prob > 0.55 and x_prob > 0.60: ...
            
    return df

def simulate_trades(df):
    """Simple PnL simulation"""
    df['win'] = False
    df['pnl'] = 0.0
    payout = 0.87
    
    trades = 0
    wins = 0
    
    # We predict row 'i'. So we check row 'i'.
    # We can iterate from 0 to len(df)-1
    for i in range(len(df)):
        sig = df['signal'].iloc[i]
        if sig != 0:
            trades += 1
            # Current candle result
            curr_c = df.iloc[i]
            is_green = curr_c['close'] > curr_c['open']
            is_red = curr_c['close'] < curr_c['open']
            
            is_win = False
            if sig == 1 and is_green: is_win = True
            if sig == -1 and is_red: is_win = True
            
            if is_win:
                df.at[i, 'win'] = True
                df.at[i, 'pnl'] = payout
                wins += 1
            else:
                df.at[i, 'pnl'] = -1.0
                
    return df

async def run_backtest_lstm():
    email = os.getenv("IQ_EMAIL")
    password = os.getenv("IQ_PASSWORD")
    api = IQOptionAPI()
    await api._connect()
    
    target_assets = ["EURUSD"] # Only testing EURUSD for now as it's the main target
    correlated_map = {
        "EURUSD": ["USDJPY", "GBPUSD", "AUDUSD"]
    }
    
    for asset in target_assets:
        correlated = correlated_map.get(asset, [])
        df = await fetch_and_prepare_multi_asset(api, asset, correlated, 300, 2000)
        
        if df is None or df.empty:
            logger.warning(f"Skipping {asset} (No Data)")
            continue
        
        df = apply_hybrid_strategy(df)
        df = simulate_trades(df)
        
        total = len(df[df['signal']!=0])
        wins = len(df[df['win']==True])
        wr = (wins/total*100) if total > 0 else 0
        pnl = df['pnl'].sum()
        
        print(f"For {asset}: Trades={total}, WR={wr:.2f}%, PnL={pnl:.2f}")
        
    api.websocket.close()

if __name__ == "__main__":
    asyncio.run(run_backtest_lstm())
