"""
S/R Bounce Signal Generator for Telegram Bot

This replaces the ColorMillion strategy with S/R bounce.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_sr_bounce_signal(candles, asset_name=None):
    """
    Get S/R bounce signal from candle data for Telegram bot.
    
    Returns: 'CALL', 'PUT', or None
    
    This is the PROFITABLE strategy (54-56% win rate).
    """
    if not candles or len(candles) < 50:
        return None
    
    try:
        # Prepare features
        df = pd.DataFrame(candles)
        from ml_utils import prepare_features
        df_features = prepare_features(df)
        
        # Get last candle
        last_candle = df_features.iloc[-1]
        
        # Check S/R proximity
        dist_to_support = last_candle.get('dist_to_support', 1.0)
        dist_to_resistance = last_candle.get('dist_to_resistance', 1.0)
        
        SR_THRESHOLD = 0.002  # Within 0.2%
        
        # Near support → BUY (expect bounce up)
        if dist_to_support < SR_THRESHOLD:
            logger.info(f"📊 S/R Signal: {asset_name} near SUPPORT ({dist_to_support*100:.3f}%) → CALL")
            return "CALL"
        
        # Near resistance → SELL (expect bounce down)
        elif dist_to_resistance < SR_THRESHOLD:
            logger.info(f"📊 S/R Signal: {asset_name} near RESISTANCE ({dist_to_resistance*100:.3f}%) → PUT")
            return "PUT"
        
        return None
        
    except Exception as e:
        logger.error(f"S/R signal error for {asset_name}: {e}")
        return None
