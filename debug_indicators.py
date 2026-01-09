"""
Debug script to check what indicators are available
"""
import pandas as pd
from ml_utils import prepare_features
from iqclient import IQOptionAPI
import asyncio

async def debug_indicators():
    # Get some sample data
    api = IQOptionAPI()
    await api._connect()
    
    candles = api.get_candle_history("BTCUSD", 1000, 300)
    
    if candles:
        # Convert to DataFrame first
        df_raw = pd.DataFrame(candles)
        
        # Prepare features
        df = prepare_features(df_raw)
        
        print("Available columns in df_features:")
        print("="*60)
        for col in sorted(df.columns):
            print(f"  - {col}")
        
        print("\n" + "="*60)
        print("Sample values from last row:")
        print("="*60)
        last_row = df.iloc[-1]
        for col in ['open', 'close', 'ema_13', 'ema_50', 'sma_20', 'sma_50', 'macd', 'macd_signal', 'adx']:
            if col in df.columns:
                print(f"{col:15} = {last_row[col]}")
            else:
                print(f"{col:15} = NOT FOUND!")
        
        # Check engulfing logic
        print("\n" + "="*60)
        print("Testing engulfing detection:")
        print("="*60)
        
        for i in range(-5, 0):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            open_curr = row['open']
            close_curr = row['close']
            open_prev = prev_row['open']
            close_prev = prev_row['close']
            
            is_green = close_curr > open_curr
            is_red = close_curr < open_curr
            prev_is_green = close_prev > open_prev
            prev_is_red = close_prev < open_prev
            
            body_curr = abs(close_curr - open_curr)
            body_prev = abs(close_prev - open_prev)
            is_engulfing = body_curr > body_prev
            
            print(f"Candle {i}: {'GREEN' if is_green else 'RED':5} | Prev: {'GREEN' if prev_is_green else 'RED':5} | Engulfing: {is_engulfing}")
    
    await api._disconnect()

if __name__ == "__main__":
    asyncio.run(debug_indicators())
