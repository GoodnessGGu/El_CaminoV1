"""
Volatility Mean Reversion Strategy

This strategy trades statistical outliers (mean reversion) using:
1. Smart Volatility Filter: 2-sigma candle body size detection
2. Engulfing Pattern: Current candle engulfs previous
3. Overextension: Bollinger Band breach or extreme RSI
4. Support/Resistance: Additional confirmation

Logic:
- CALL: Massive RED engulfing candle at support/oversold → Snap-back up
- PUT: Massive GREEN engulfing candle at resistance/overbought → Drawback down

Expiry: 1 candle (matches the timeframe being analyzed)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Strategy Parameters
LOOKBACK_PERIOD = 20        # Candles for statistical analysis
SIGMA_THRESHOLD = 2.0       # Standard deviations from mean
BB_PERIOD = 20              # Bollinger Band period
BB_STD = 2                  # Bollinger Band standard deviations
RSI_PERIOD = 14             # RSI period
RSI_OVERSOLD = 30           # RSI oversold threshold
RSI_OVERBOUGHT = 70         # RSI overbought threshold


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    return df


def calculate_rsi(df, period=14):
    """Calculate RSI."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calculate_support_resistance(df, lookback=50):
    """Calculate support and resistance levels."""
    df['support'] = df['min'].rolling(window=lookback).min()
    df['resistance'] = df['max'].rolling(window=lookback).max()
    
    # Distance to S/R as percentage
    df['dist_to_support'] = abs(df['close'] - df['support']) / df['close']
    df['dist_to_resistance'] = abs(df['close'] - df['resistance']) / df['close']
    
    return df


def get_volatility_mean_reversion_signal(candles, asset_name=None):
    """
    Volatility-based mean reversion strategy.
    
    Returns: 'CALL', 'PUT', or None
    
    Trades statistical outliers (2-sigma moves) that are overextended.
    """
    if not candles or len(candles) < LOOKBACK_PERIOD + 10:
        return None
    
    try:
        # Prepare DataFrame
        df = pd.DataFrame(candles)
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        # Calculate indicators
        df = calculate_bollinger_bands(df, BB_PERIOD, BB_STD)
        df = calculate_rsi(df, RSI_PERIOD)
        df = calculate_support_resistance(df, lookback=50)
        
        # Get last 2 candles
        if len(df) < 2:
            return None
            
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for NaN values
        required_cols = ['bb_upper', 'bb_lower', 'rsi', 'support', 'resistance']
        if any(pd.isna(curr[col]) for col in required_cols):
            return None
        
        # === 1. SMART VOLATILITY FILTER (Statistical Outlier Detection) ===
        
        # Calculate candle bodies for last LOOKBACK_PERIOD candles
        df['candle_body'] = abs(df['close'] - df['open'])
        recent_bodies = df['candle_body'].iloc[-(LOOKBACK_PERIOD+1):-1]  # Exclude current
        
        mean_size = recent_bodies.mean()
        std_dev = recent_bodies.std()
        
        current_body = abs(curr['close'] - curr['open'])
        prev_body = abs(prev['close'] - prev['open'])
        
        # Is this candle a statistical outlier? (2-sigma from mean)
        is_statistically_massive = current_body > (mean_size + (SIGMA_THRESHOLD * std_dev))
        
        # === 2. PATTERN DETECTION ===
        
        # Candle colors
        is_green = curr['close'] > curr['open']
        is_red = curr['close'] < curr['open']
        
        # Engulfing pattern
        is_engulfing = current_body > prev_body
        
        # === 3. OVER-EXTENSION CHECK ===
        
        # Bollinger Band breach
        price_above_upper_bb = curr['close'] > curr['bb_upper']
        price_below_lower_bb = curr['close'] < curr['bb_lower']
        
        # RSI extremes
        rsi_overbought = curr['rsi'] > RSI_OVERBOUGHT
        rsi_oversold = curr['rsi'] < RSI_OVERSOLD
        
        # Support/Resistance proximity (within 0.3%)
        at_resistance = curr['dist_to_resistance'] < 0.003
        at_support = curr['dist_to_support'] < 0.003
        
        # Combined overextension
        is_overextended_up = price_above_upper_bb or rsi_overbought or at_resistance
        is_overextended_down = price_below_lower_bb or rsi_oversold or at_support
        
        # === 4. SIGNAL GENERATION ===
        
        # PUT Signal: Massive GREEN engulfing candle that's overextended
        # Logic: Market exhausted after big green move → Expect drawback
        if is_green and is_engulfing and is_statistically_massive:
            if is_overextended_up:
                reason = []
                if price_above_upper_bb:
                    reason.append("Above BB")
                if rsi_overbought:
                    reason.append(f"RSI {curr['rsi']:.1f}")
                if at_resistance:
                    reason.append("At Resistance")
                
                logger.info(f"📊 Mean Reversion: {asset_name} → PUT (Drawback)")
                logger.info(f"   Massive GREEN candle (2σ outlier), Overextended: {', '.join(reason)}")
                logger.info(f"   Body: {current_body:.5f} vs Mean: {mean_size:.5f} (Threshold: {mean_size + SIGMA_THRESHOLD*std_dev:.5f})")
                return "PUT"
        
        # CALL Signal: Massive RED engulfing candle that's overextended
        # Logic: Sellers exhausted after big red move → Expect snap-back
        elif is_red and is_engulfing and is_statistically_massive:
            if is_overextended_down:
                reason = []
                if price_below_lower_bb:
                    reason.append("Below BB")
                if rsi_oversold:
                    reason.append(f"RSI {curr['rsi']:.1f}")
                if at_support:
                    reason.append("At Support")
                
                logger.info(f"📊 Mean Reversion: {asset_name} → CALL (Snap-back)")
                logger.info(f"   Massive RED candle (2σ outlier), Overextended: {', '.join(reason)}")
                logger.info(f"   Body: {current_body:.5f} vs Mean: {mean_size:.5f} (Threshold: {mean_size + SIGMA_THRESHOLD*std_dev:.5f})")
                return "CALL"
        
        return None
        
    except Exception as e:
        logger.error(f"Volatility mean reversion error for {asset_name}: {e}")
        return None


def get_volatility_stats(candles, asset_name=None):
    """
    Get volatility statistics for debugging/analysis.
    Returns dict with current candle stats vs historical mean.
    """
    if not candles or len(candles) < LOOKBACK_PERIOD + 10:
        return None
    
    try:
        df = pd.DataFrame(candles)
        cols = ['open', 'close', 'min', 'max']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        df['candle_body'] = abs(df['close'] - df['open'])
        recent_bodies = df['candle_body'].iloc[-(LOOKBACK_PERIOD+1):-1]
        
        mean_size = recent_bodies.mean()
        std_dev = recent_bodies.std()
        current_body = abs(df.iloc[-1]['close'] - df.iloc[-1]['open'])
        
        sigma_distance = (current_body - mean_size) / std_dev if std_dev > 0 else 0
        
        return {
            'asset': asset_name,
            'current_body': current_body,
            'mean_body': mean_size,
            'std_dev': std_dev,
            'sigma_distance': sigma_distance,
            'is_outlier': sigma_distance > SIGMA_THRESHOLD,
            'threshold': mean_size + (SIGMA_THRESHOLD * std_dev)
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return None
