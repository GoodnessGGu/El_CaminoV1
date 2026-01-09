"""
Normal S/R Bounce Strategy (NOT Reversed)

This is the PROFITABLE strategy for the Telegram bot.
Use this for actual trading.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def label_sr_bounces(df_features):
    """
    Label trades using NORMAL S/R Bounce strategy.
    
    This is the profitable strategy (54-58% win rate).
    
    Signals when:
    - Price near support (within 0.2%) → BUY (expect bounce up)
    - Price near resistance (within 0.2%) → SELL (expect bounce down)
    
    Args:
        df_features: DataFrame with OHLCV data and features
        
    Returns:
        DataFrame with labeled training examples
    """
    logger.info("Labeling data using S/R Bounce Strategy...")
    
    records = df_features.to_dict('records')
    labeled_data = []
    
    min_candles = 50
    SR_THRESHOLD = 0.002  # Within 0.2% of S/R level
    
    for i in range(min_candles, len(records) - 1):
        row = records[i]
        next_candle = records[i+1]
        
        # Get price and S/R levels
        close = row.get('close')
        support = row.get('support_50')
        resistance = row.get('resistance_50')
        dist_to_support = row.get('dist_to_support')
        dist_to_resistance = row.get('dist_to_resistance')
        
        # Skip if missing data
        if None in [close, support, resistance, dist_to_support, dist_to_resistance]:
            continue
        
        signal = None
        entry_reason = None
        
        # === Near Support → BUY (expect bounce up) ===
        if dist_to_support < SR_THRESHOLD:
            signal = "CALL"
            entry_reason = "sr_bounce_support"
            # Win if price goes UP
            is_win = next_candle['close'] > close
        
        # === Near Resistance → SELL (expect bounce down) ===
        elif dist_to_resistance < SR_THRESHOLD:
            signal = "PUT"
            entry_reason = "sr_bounce_resistance"
            # Win if price goes DOWN
            is_win = next_candle['close'] < close
        
        # If we have a signal, add to labeled data
        if signal:
            row['signal'] = signal
            row['outcome'] = 1 if is_win else 0
            row['entry_reason'] = entry_reason
            labeled_data.append(row)
    
    if not labeled_data:
        logger.warning(f"No S/R bounce signals found in {len(df_features)} candles.")
        return None
    
    logger.info(f"Found {len(labeled_data)} S/R bounce training examples.")
    
    # Calculate and log win rate
    df_labeled = pd.DataFrame(labeled_data)
    win_rate = df_labeled['outcome'].mean()
    logger.info(f"📊 S/R Bounce Strategy Win Rate: {win_rate:.2%}")
    
    # Log signal distribution
    call_count = (df_labeled['signal'] == 'CALL').sum()
    put_count = (df_labeled['signal'] == 'PUT').sum()
    logger.info(f"   CALL signals (support bounces): {call_count} ({call_count/len(df_labeled):.1%})")
    logger.info(f"   PUT signals (resistance bounces): {put_count} ({put_count/len(df_labeled):.1%})")
    
    return df_labeled
