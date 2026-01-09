"""
Advanced Feature Engineering Module

Adds market context and order flow features to improve ML model performance.
These features help the model understand market regime and price action dynamics.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_market_context_features(df):
    """
    Adds features that capture market regime and context.
    
    Features added:
    - Market regime (trending vs ranging)
    - Volatility regime (high vs low)
    - Time of day (trading sessions)
    - Support/Resistance levels
    - Momentum score
    - Volume profile
    
    Args:
        df: DataFrame with OHLCV data and basic indicators (RSI, ADX, ATR, etc.)
        
    Returns:
        DataFrame: Original df with additional market context features
    """
    df = df.copy()
    
    # 1. Market Regime (Trending vs Ranging)
    # Use ADX: >25 = Trending, <20 = Ranging
    if 'adx' in df.columns:
        df['regime_trending'] = (df['adx'] > 25).astype(int)
        df['regime_ranging'] = (df['adx'] < 20).astype(int)
    else:
        df['regime_trending'] = 0
        df['regime_ranging'] = 0
    
    # 2. Volatility Regime (High vs Low)
    # Use ATR percentile over rolling window
    if 'atr' in df.columns:
        df['atr_percentile'] = df['atr'].rolling(100, min_periods=20).apply(
            lambda x: (x.iloc[-1] >= x.quantile(0.7)).astype(int) if len(x) > 0 else 0
        )
    else:
        df['atr_percentile'] = 0
    
    # 3. Time of Day (for non-OTC markets)
    if 'time' in df.columns:
        try:
            # Ensure time is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            df['hour'] = df['time'].dt.hour
            
            # Trading sessions (UTC times)
            # London: 8:00-16:00 UTC
            # New York: 13:00-21:00 UTC
            # Overlap: 13:00-16:00 UTC
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        except Exception as e:
            logger.warning(f"Could not extract time features: {e}")
            df['hour'] = 0
            df['is_london_session'] = 0
            df['is_ny_session'] = 0
            df['is_overlap'] = 0
    else:
        df['hour'] = 0
        df['is_london_session'] = 0
        df['is_ny_session'] = 0
        df['is_overlap'] = 0
    
    # 4. Support/Resistance Levels
    # Simple approach: Rolling min/max
    if 'max' in df.columns and 'min' in df.columns and 'close' in df.columns:
        df['resistance_50'] = df['max'].rolling(50, min_periods=10).max()
        df['support_50'] = df['min'].rolling(50, min_periods=10).min()
        
        # Distance to S/R as percentage
        df['dist_to_resistance'] = ((df['resistance_50'] - df['close']) / df['close'] * 100).fillna(0)
        df['dist_to_support'] = ((df['close'] - df['support_50']) / df['close'] * 100).fillna(0)
    else:
        df['resistance_50'] = 0
        df['support_50'] = 0
        df['dist_to_resistance'] = 0
        df['dist_to_support'] = 0
    
    # 5. Momentum Strength Score
    # Combine RSI + MACD histogram for overall momentum
    if 'rsi' in df.columns and 'macd_hist' in df.columns and 'atr' in df.columns:
        # Normalize RSI to [-1, 1] range
        rsi_normalized = (df['rsi'] - 50) / 50
        
        # Normalize MACD histogram using ATR
        macd_normalized = np.sign(df['macd_hist']) * np.minimum(
            abs(df['macd_hist']) / (df['atr'] + 1e-8), 1
        )
        
        # Combined momentum score
        df['momentum_score'] = (rsi_normalized + macd_normalized) / 2
    else:
        df['momentum_score'] = 0
    
    # 6. Volume Profile (for non-OTC assets only)
    # OTC has fake volume, so skip for OTC assets
    if 'volume' in df.columns:
        # Check if this is OTC data (volume will be very uniform/fake)
        volume_std = df['volume'].std()
        volume_mean = df['volume'].mean()
        
        # If volume varies significantly, it's likely real
        if volume_mean > 0 and (volume_std / volume_mean) > 0.1:
            df['volume_ma'] = df['volume'].rolling(20, min_periods=5).mean()
            df['volume_ratio'] = (df['volume'] / (df['volume_ma'] + 1e-8)).fillna(1.0)
        else:
            # OTC or no real volume data
            df['volume_ratio'] = 1.0
    else:
        df['volume_ratio'] = 1.0
    
    logger.info(f"✅ Added {13} market context features")
    
    return df


def add_order_flow_features(df):
    """
    Adds features based on price action and order flow.
    
    Features added:
    - Buying/Selling pressure
    - Net pressure (cumulative)
    - Wick rejection analysis
    - Consecutive candle streaks
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame: Original df with additional order flow features
    """
    df = df.copy()
    
    if 'open' not in df.columns or 'close' not in df.columns:
        logger.warning("Missing OHLC data for order flow features")
        return df
    
    # 1. Buying/Selling Pressure
    # Green candles = Buying, Red = Selling
    is_green = (df['close'] > df['open']).astype(int)
    is_red = (df['close'] < df['open']).astype(int)
    
    df['buying_pressure'] = is_green * abs(df['close'] - df['open'])
    df['selling_pressure'] = is_red * abs(df['close'] - df['open'])
    
    # 2. Net Pressure (cumulative over 10 candles)
    df['net_pressure_10'] = (
        df['buying_pressure'].rolling(10, min_periods=1).sum() - 
        df['selling_pressure'].rolling(10, min_periods=1).sum()
    )
    
    # 3. Wick Rejection Analysis
    # Long upper wick = Rejection of higher prices (bearish)
    # Long lower wick = Rejection of lower prices (bullish)
    if 'max' in df.columns and 'min' in df.columns:
        df['upper_wick'] = df['max'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['min']
        
        # Wick bias: positive = bullish (lower wick dominant), negative = bearish
        total_range = df['max'] - df['min']
        df['wick_bias'] = ((df['lower_wick'] - df['upper_wick']) / (total_range + 1e-8)).fillna(0)
    else:
        df['upper_wick'] = 0
        df['lower_wick'] = 0
        df['wick_bias'] = 0
    
    # 4. Consecutive Candles (Streak Counter)
    # Positive streak = consecutive green, negative = consecutive red
    df['candle_color'] = is_green.astype(int) * 2 - 1  # 1=green, -1=red
    
    # Calculate streak
    df['streak'] = 0
    streak = 0
    prev_color = 0
    
    for i in range(len(df)):
        color = df['candle_color'].iloc[i]
        if color == prev_color:
            streak += 1
        else:
            streak = 1
        df.at[df.index[i], 'streak'] = streak * color  # Positive=green streak, negative=red
        prev_color = color
    
    logger.info(f"✅ Added {8} order flow features")
    
    return df


def add_all_advanced_features(df):
    """
    Convenience function to add all advanced features at once.
    
    Args:
        df: DataFrame with OHLCV data and basic indicators
        
    Returns:
        DataFrame: df with all advanced features added
    """
    df = add_market_context_features(df)
    df = add_order_flow_features(df)
    
    return df
