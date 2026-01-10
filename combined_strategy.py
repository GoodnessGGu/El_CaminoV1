"""
Combined ColorMillion + Engulfing Strategy

This strategy combines two proven approaches:
1. ColorMillion: EMA13, MACD momentum with trend filter
2. Script.txt Engulfing: SMA 3, 7, 200 with engulfing patterns

Signal Generation:
- CALL: Both strategies agree on bullish signal
- PUT: Both strategies agree on bearish signal
- None: Strategies disagree or no signal
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_combined_signal(candles, asset_name=None):
    """
    Combined ColorMillion + Engulfing strategy for Telegram bot.
    
    Returns: 'CALL', 'PUT', or None
    
    This combines:
    - ColorMillion: EMA13 + MACD momentum
    - Engulfing: SMA 3/7/200 + engulfing patterns
    """
    if not candles or len(candles) < 200:
        return None
    
    try:
        # Prepare DataFrame
        df = pd.DataFrame(candles)
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        # === PART 1: ColorMillion Indicators ===
        
        # EMA 13
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        
        # MACD (12, 26, 9)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema_12 - ema_26
        df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        
        # EMA 200 (Trend Filter)
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # ADX (Strength Filter)
        high = df['max']
        low = df['min']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # === PART 2: Engulfing Pattern Indicators ===
        
        # SMA 3, 7, 200
        df['sma_3'] = df['close'].rolling(window=3).mean()
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Get last 2 candles
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for NaN values
        required_cols = ['ema_13', 'macd_hist', 'ema_200', 'sma_3', 'sma_7', 'sma_200', 'adx']
        if any(pd.isna(curr[col]) for col in required_cols):
            return None
        
        # === STRATEGY 1: ColorMillion Signal ===
        colormillion_signal = None
        
        ema_curr = curr['ema_13']
        ema_prev = prev['ema_13']
        hist_curr = curr['macd_hist']
        hist_prev = prev['macd_hist']
        ema_200 = curr['ema_200']
        adx = curr['adx']
        close_curr = curr['close']
        
        MIN_ADX = 18  # Lowered from 20 for more signals
        
        # ColorMillion CALL: EMA13 rising + MACD rising + above EMA200 + ADX strong
        if ema_curr > ema_prev and hist_curr > hist_prev:
            if close_curr > ema_200 and adx > MIN_ADX:
                colormillion_signal = "CALL"
        
        # ColorMillion PUT: EMA13 falling + MACD falling + below EMA200 + ADX strong
        elif ema_curr < ema_prev and hist_curr < hist_prev:
            if close_curr < ema_200 and adx > MIN_ADX:
                colormillion_signal = "PUT"
        
        # === STRATEGY 2: Engulfing Pattern Signal ===
        engulfing_signal = None
        
        open_curr = curr['open']
        close_curr = curr['close']
        open_prev = prev['open']
        close_prev = prev['close']
        
        sma_3 = curr['sma_3']
        sma_7 = curr['sma_7']
        sma_200 = curr['sma_200']
        
        # Candle colors
        is_green = close_curr > open_curr
        prev_is_red = close_prev < open_prev
        is_red = close_curr < open_curr
        prev_is_green = close_prev > open_prev
        
        # Engulfing patterns
        is_bullish_engulfing = (is_green and prev_is_red and 
                                close_curr > open_prev and 
                                open_curr <= close_prev)
        
        is_bearish_engulfing = (is_red and prev_is_green and 
                                close_curr < open_prev and 
                                open_curr >= close_prev)
        
        # Engulfing CALL: Bullish engulfing + above SMAs
        if is_bullish_engulfing:
            if close_curr > sma_3 and close_curr > sma_7 and close_curr > sma_200:
                engulfing_signal = "CALL"
        
        # Engulfing PUT: Bearish engulfing + below SMAs
        elif is_bearish_engulfing:
            if close_curr < sma_3 and close_curr < sma_7 and close_curr < sma_200:
                engulfing_signal = "PUT"
        
        # === COMBINE SIGNALS ===
        # Only trade when BOTH strategies agree
        if colormillion_signal == "CALL" and engulfing_signal == "CALL":
            logger.info(f"📊 Combined Signal: {asset_name} → CALL (ColorMillion + Engulfing)")
            return "CALL"
        
        elif colormillion_signal == "PUT" and engulfing_signal == "PUT":
            logger.info(f"📊 Combined Signal: {asset_name} → PUT (ColorMillion + Engulfing)")
            return "PUT"
        
        # Log when strategies disagree (for debugging)
        if colormillion_signal and engulfing_signal and colormillion_signal != engulfing_signal:
            logger.debug(f"⚠️ {asset_name}: Strategies disagree (CM: {colormillion_signal}, ENG: {engulfing_signal})")
        
        return None
        
    except Exception as e:
        logger.error(f"Combined strategy error for {asset_name}: {e}")
        return None


def get_colormillion_only(candles, asset_name=None):
    """
    ColorMillion strategy only (for comparison/testing).
    Returns: 'CALL', 'PUT', or None
    """
    if not candles or len(candles) < 200:
        return None
    
    try:
        df = pd.DataFrame(candles)
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        # EMA 13
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema_12 - ema_26
        df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        
        # EMA 200
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # ADX
        high = df['max']
        low = df['min']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if pd.isna(curr['ema_13']) or pd.isna(curr['macd_hist']) or pd.isna(curr['ema_200']):
            return None
        
        ema_curr = curr['ema_13']
        ema_prev = prev['ema_13']
        hist_curr = curr['macd_hist']
        hist_prev = prev['macd_hist']
        ema_200 = curr['ema_200']
        adx = curr['adx']
        close_curr = curr['close']
        
        MIN_ADX = 18
        
        if ema_curr > ema_prev and hist_curr > hist_prev:
            if close_curr > ema_200 and adx > MIN_ADX:
                logger.info(f"📊 ColorMillion: {asset_name} → CALL")
                return "CALL"
        
        elif ema_curr < ema_prev and hist_curr < hist_prev:
            if close_curr < ema_200 and adx > MIN_ADX:
                logger.info(f"📊 ColorMillion: {asset_name} → PUT")
                return "PUT"
        
        return None
        
    except Exception as e:
        logger.error(f"ColorMillion error for {asset_name}: {e}")
        return None


def get_engulfing_only(candles, asset_name=None):
    """
    Engulfing pattern strategy only (for comparison/testing).
    Returns: 'CALL', 'PUT', or None
    """
    if not candles or len(candles) < 200:
        return None
    
    try:
        df = pd.DataFrame(candles)
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        # SMA 3, 7, 200
        df['sma_3'] = df['close'].rolling(window=3).mean()
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if pd.isna(curr['sma_3']) or pd.isna(curr['sma_7']) or pd.isna(curr['sma_200']):
            return None
        
        open_curr = curr['open']
        close_curr = curr['close']
        open_prev = prev['open']
        close_prev = prev['close']
        
        sma_3 = curr['sma_3']
        sma_7 = curr['sma_7']
        sma_200 = curr['sma_200']
        
        is_green = close_curr > open_curr
        prev_is_red = close_prev < open_prev
        is_red = close_curr < open_curr
        prev_is_green = close_prev > open_prev
        
        is_bullish_engulfing = (is_green and prev_is_red and 
                                close_curr > open_prev and 
                                open_curr <= close_prev)
        
        is_bearish_engulfing = (is_red and prev_is_green and 
                                close_curr < open_prev and 
                                open_curr >= close_prev)
        
        if is_bullish_engulfing:
            if close_curr > sma_3 and close_curr > sma_7 and close_curr > sma_200:
                logger.info(f"📊 Engulfing: {asset_name} → CALL (Bullish Engulfing)")
                return "CALL"
        
        elif is_bearish_engulfing:
            if close_curr < sma_3 and close_curr < sma_7 and close_curr < sma_200:
                logger.info(f"📊 Engulfing: {asset_name} → PUT (Bearish Engulfing)")
                return "PUT"
        
        return None
        
    except Exception as e:
        logger.error(f"Engulfing error for {asset_name}: {e}")
        return None
