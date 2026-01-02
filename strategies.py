import pandas as pd
import numpy as np
import logging
from settings import config

logger = logging.getLogger(__name__)

# --- Pattern Recognition Configuration (Dummy for imports) ---
PATTERN_CONFIG = {
    'engulfing': True,
    'pinbar': False,
    'marubozu': False
}

def analyze_strategy(candles_data, asset_name=None):
    """
    Analyzes candle data using ONLY the 'New Script' logic (SMA 3, 7, 200 + Engulfing).
    No AI, No ML features, No heavy processing.
    
    Args:
        candles_data: List of candle dicts
        asset_name: Optional asset name (e.g. "EURUSD") to Determine Real vs OTC logic.
    """
    if not candles_data or len(candles_data) < 200:
        # Need enough data for SMA 200
        return None

    try:
        # Create DataFrame
        df = pd.DataFrame(candles_data)
        
        # Standardize Columns
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        # Calculate Indicators (Lightweight)
        # Strategy: 
        # Fast MAP = 3, Slow MAP = 7, Trend MAP = 200
        # CALL: Green Candle, Prev Red, Engulfing, Close > SMA3 & SMA7 & SMA200
        # PUT: Red Candle, Prev Green, Engulfing, Close < SMA3 & SMA7 & SMA200
        
        df['sma_3'] = df['close'].rolling(window=3).mean()
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Get last 2 rows
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Validate SMAs exist (not NaN)
        if pd.isna(curr['sma_200']):
            return None
            
        # Parse Values
        close_curr = curr['close']
        open_curr = curr['open']
        close_prev = prev['close']
        open_prev = prev['open']
        
        sma3 = curr['sma_3']
        sma7 = curr['sma_7']
        sma200 = curr['sma_200']
        
        # Candle Colors
        is_green = close_curr > open_curr
        is_red = close_curr < open_curr
        prev_is_green = close_prev > open_prev
        prev_is_red = close_prev < open_prev
        
        # Engulfing (Body Size)
        body_curr = abs(close_curr - open_curr)
        body_prev = abs(close_prev - open_prev)
        is_engulfing = body_curr > body_prev
        
        # --- VOLUME VALIDATION (REAL MARKETS) ---
        is_volume_valid = True
        if asset_name and "-OTC" not in asset_name and "OTC" not in asset_name:
             # Logic: Current Volume > Previous Volume (Momentum confirmation)
             vol_curr = curr.get('volume', 0)
             vol_prev = prev.get('volume', 0)
             if vol_curr <= vol_prev:
                 is_volume_valid = False
                 # logger.info(f"Filtered {asset_name}: Low Volume Breakout")
        
        # Check Strategy Conditions
        signal = None
        
        if is_engulfing and is_volume_valid:
            # CALL SIGNAL
            # Logic: Green, Prev Red, Engulfing, Above ALL SMAs
            if is_green and prev_is_red:
                if close_curr > sma3 and close_curr > sma7 and close_curr > sma200:
                    signal = "CALL"
            
            # PUT SIGNAL
            # Logic: Red, Prev Green, Engulfing, Below ALL SMAs
            elif is_red and prev_is_green:
                if close_curr < sma3 and close_curr < sma7 and close_curr < sma200:
                    signal = "PUT"
                
        return signal

    except Exception as e:
        logger.error(f"Strategy Error: {e}")
        return None

def confirm_trade_with_ai(candles_data, direction):
    """
    Dummy function to satisfy imports. 
    Always returns True since AI is disabled for this strategy.
    """
    return True

def reload_ai_model():
    # Placeholder to prevent import errors in main
    pass

def analyze_colormillion(candles_data, asset_name=None):
    """
    Analyzes candle data using 'ColorMillion 2023' Logic.
    
    Indicators:
    - EMA 13
    - MACD (12, 26, 9)
    
    Signal Logic:
    - CALL: EMA13 > Prev EMA13  AND  MACD_Hist > Prev MACD_Hist
    - PUT:  EMA13 < Prev EMA13  AND  MACD_Hist < Prev MACD_Hist
    
    Args:
        candles_data: List of candle dicts
        asset_name: Optional, for consistency/logging
    """
    if not candles_data or len(candles_data) < 50:
         return None

    try:
        df = pd.DataFrame(candles_data)
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])

        # --- 1. EMA 13 ---
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()

        # --- 2. MACD (12, 26, 9) ---
        # Fast EMA 12
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        # Slow EMA 26
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd_line'] = ema_12 - ema_26
        df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']

        # --- 3. EMA 200 (Trend Filter) ---
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # --- 4. ADX (Volatility Filter) ---
        # Simplified ADX implementation
        high = df['max']
        low = df['min']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # Get Last 2 Candles
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check if indicators are valid (not NaN)
        # Ensure we have enough data for EMA 200
        if pd.isna(curr['ema_13']) or pd.isna(curr['macd_hist']) or pd.isna(curr['ema_200']):
            return None

        ema_curr = curr['ema_13']
        ema_prev = prev['ema_13']
        
        hist_curr = curr['macd_hist']
        hist_prev = prev['macd_hist']
        
        ema_200 = curr['ema_200']
        adx = curr.get('adx', 0)
        close_curr = curr['close']
        
        signal = None
        
        # Filter Settings
        MIN_ADX = 20
        
        # CALL Logic: Both Rising + Above EMA 200 + ADX OK
        if ema_curr > ema_prev and hist_curr > hist_prev:
            if close_curr > ema_200 and adx > MIN_ADX:
                signal = "CALL"
            elif close_curr <= ema_200:
                # logger.info(f"Filtered CALL on {asset_name}: Price below EMA 200")
                pass
            
        # PUT Logic: Both Falling + Below EMA 200 + ADX OK
        elif ema_curr < ema_prev and hist_curr < hist_prev:
            if close_curr < ema_200 and adx > MIN_ADX:
                signal = "PUT"
            elif close_curr >= ema_200:
                # logger.info(f"Filtered PUT on {asset_name}: Price above EMA 200")
                pass
            
        return signal

    except Exception as e:
        logger.error(f"ColorMillion Error: {e}")
        return None
