import logging
import time
import pandas as pd
import numpy as np
import os
import asyncio
from iqclient import IQOptionAPI
from dotenv import load_dotenv
from ml_utils import prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
EMAIL = os.getenv("IQ_EMAIL")
PASSWORD = os.getenv("IQ_PASSWORD")

if not EMAIL or not PASSWORD:
    logger.error("❌ Credentials not found. Please set EMAIL and PASSWORD in .env file.")
    exit(1)

API = IQOptionAPI(EMAIL, PASSWORD)

def connect_iq():
    # Run async connection
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connected = loop.run_until_complete(API.ensure_connect())
    
    if connected:
        logger.info("✅ Connected to IQ Option successfully (via iqclient).")
        # Give it a moment to stabilize
        time.sleep(2)
    else:
        logger.error(f"❌ Connection failed.")
        exit(1)

def get_candles(asset, timeframe=60, amount=1000, endtime=None):
    """Fetches candles from IQ Option."""
    if endtime is None:
        endtime = int(time.time())
    else:
        endtime = int(endtime)

    try:
        # iqclient.get_candle_history args: asset_name, count, timeframe, end_time
        candles = API.get_candle_history(asset, count=amount, timeframe=timeframe, end_time=endtime)
        return candles
    except Exception as e:
        logger.error(f"Error fetching candles for {asset}: {e}")
        return []

def label_data_binary_strategy(df):
    """
    Labels data based on a simple strategy or just Next Candle Color.
    For Training, simpler is often better: 
    Let's label: If next candle close > current close -> CALL WIN.
    
    Actually, to train a model to finding "Winning Conditions" for a specific strategy, 
    we usually need that strategy's triggers.
    
    However, a general "Next Candle Predictor" is also useful.
    Let's create a dataset where Outcome = 1 if the *direction matches the prediction*.
    
    But we don't have predictions yet.
    
    Standard Approach: 
    Target = 1 if Next Candle is GREEN (Close > Open).
    Target = 0 if Next Candle is RED (Close < Open).
    
    Features = Current Candle Indicators.
    
    The model will learn: "When RSI is low, Next Candle is likely Green".
    """
    
    # 1. Target: Next Candle Direction
    # Shift(-1) means looking into the future (next row)
    df['next_close'] = df['close'].shift(-1)
    df['next_open'] = df['open'].shift(-1)
    
    # Define Win Condition: Green Candle
    # If Next Close > Next Open => 1 (CALL)
    # If Next Close < Next Open => 0 (PUT)
    
    # But wait, we want to train for BOTH Call and Put?
    # Usually we train one model for "Is this a WIN?".
    # So we need to know what the signal WAS.
    
    # Let's simplify: Train the model to predict "Will the next candle go UP?".
    # > 0.5 = UP (Call), < 0.5 = DOWN (Put).
    # Then in the bot: If Model > 0.6 -> Approve Call. If Model < 0.4 -> Approve Put.
    
    # Let's try this:
    # outcome = 1 (Green), 0 (Red)
    conditions = [
        (df['next_close'] > df['next_open']), # Green
        (df['next_close'] <= df['next_open']) # Red
    ]
    choices = [1, 0]
    df['outcome'] = np.select(conditions, choices, default=np.nan)
    
    # Drop last row (NaN because of shift)
    df = df.dropna()
    
    return df

def collect_data():
    connect_iq()
    
    # Target and Correlations
    target_asset = "EURUSD"
    correlated_assets = ["USDJPY", "GBPUSD", "AUDUSD"] # Assets to use as features
    
    all_assets = [target_asset] + correlated_assets
    asset_dfs = {}
    
    TOTAL_CANDLES = 50000 
    
    # 1. Fetch Data for ALL assets
    for asset in all_assets:
        logger.info(f"Fetching data for {asset}...")
        
        asset_candles = []
        end_time = time.time()
        
        while len(asset_candles) < TOTAL_CANDLES:
            candles = get_candles(asset, timeframe=300, amount=1000, endtime=end_time)
            
            if not candles:
                logger.warning(f"No more candles for {asset}")
                break
            
            asset_candles = candles + asset_candles
            new_end_time = candles[0]['from']
            
            if new_end_time >= end_time:
                 break
            end_time = new_end_time
            
            logger.info(f"  {asset}: {len(asset_candles)} / {TOTAL_CANDLES}")
            time.sleep(0.2)
            
        # Create DataFrame
        df = pd.DataFrame(asset_candles)
        if df.empty:
            logger.error(f"Failed to fetch data for {asset}")
            return

        df = df.drop_duplicates(subset=['from']).sort_values(by='from').reset_index(drop=True)
        
        # Standardize time
        if 'from' in df.columns:
            df['time'] = pd.to_datetime(df['from'], unit='s')
            
        # Calculate Basic Indicators per Asset (RSI, etc)
        # We need these features for the correlated assets too!
        # But prepare_features might drop 'time' or 'from', so be careful.
        # We want to keep 'time' for merging.
        
        # Let's simple rename raw columns for correlated assets
        if asset != target_asset:
            # Keep only useful columns for correlation: Close, Open, Min, Max, Volume
            cols_to_keep = ['time', 'open', 'close', 'min', 'max', 'volume']
            df = df[cols_to_keep]
            
            # Rename columns: e.g. open -> USDJPY_open
            rename_map = {c: f"{asset}_{c}" for c in df.columns if c != 'time'}
            df = df.rename(columns=rename_map)
            
        asset_dfs[asset] = df
        
    # 2. Merge DataFrames on 'time'
    logger.info("Merging Correlation Data...")
    final_df = asset_dfs[target_asset]
    
    for asset in correlated_assets:
        if asset in asset_dfs:
            # Inner join to ensure we only train on rows where we have ALL data (synchronous)
            final_df = pd.merge(final_df, asset_dfs[asset], on='time', how='inner')
            
    logger.info(f"Merged Data Shape: {final_df.shape}")
    
    # 3. Feature Engineering (Target Asset)
    # The 'prepare_features' might expect specific column names (open, close...) which EURUSD still has.
    final_df = prepare_features(final_df)
    
    # 4. Generate Target Label (Outcome)
    final_df = label_data_binary_strategy(final_df)
    
    # 5. Save
    final_df['asset'] = target_asset
    final_df.to_csv("training_data.csv", index=False)
    logger.info(f"✅ Saved Wide-Dataset to training_data.csv ({len(final_df)} rows)")

if __name__ == "__main__":
    collect_data()
