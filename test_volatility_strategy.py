"""
Test script for Volatility Mean Reversion Strategy

Tests the statistical outlier detection and mean reversion logic.
"""

import asyncio
from iqclient import IQOptionAPI
from volatility_mean_reversion_strategy import (
    get_volatility_mean_reversion_signal,
    get_volatility_stats
)

async def test_volatility_strategy():
    """Test volatility mean reversion strategy on live data."""
    
    # Initialize API
    api = IQOptionAPI()
    await api._connect()
    
    print("=" * 80)
    print("TESTING VOLATILITY MEAN REVERSION STRATEGY")
    print("=" * 80)
    print("\nStrategy Logic:")
    print("  1. Detect 2-sigma candle body outliers (statistically massive moves)")
    print("  2. Confirm with engulfing pattern")
    print("  3. Check for overextension (Bollinger Bands, RSI, S/R)")
    print("  4. Trade the reversion (snap-back or drawback)")
    
    # Test assets
    test_assets = [
        "EURUSD-OTC",
        "GBPUSD-OTC",
        "USDJPY-OTC",
        "AUDUSD-OTC",
        "BTCUSD"
    ]
    
    for asset in test_assets:
        print(f"\n{'='*80}")
        print(f"📊 Testing {asset}")
        print("=" * 80)
        
        # Fetch 1-minute candles
        candles = api.get_candle_history(asset, 300, 60)
        
        if not candles:
            print(f"❌ No candles for {asset}")
            continue
        
        # Get volatility statistics
        stats = get_volatility_stats(candles, asset)
        
        if stats:
            print(f"\n📈 Volatility Statistics:")
            print(f"   Current Body:  {stats['current_body']:.5f}")
            print(f"   Mean Body:     {stats['mean_body']:.5f}")
            print(f"   Std Dev:       {stats['std_dev']:.5f}")
            print(f"   Sigma Distance: {stats['sigma_distance']:.2f}σ")
            print(f"   Threshold (2σ): {stats['threshold']:.5f}")
            print(f"   Is Outlier:    {'✅ YES' if stats['is_outlier'] else '❌ No'}")
        
        # Get signal
        signal = get_volatility_mean_reversion_signal(candles, asset)
        
        print(f"\n🎯 Signal: {signal or 'None'}")
        
        if signal:
            print(f"\n✅ TRADE SIGNAL FOUND!")
            print(f"   Direction: {signal}")
            print(f"   Strategy: Mean Reversion (Statistical Outlier)")
            print(f"   Expiry: 1 candle (1 minute)")
        else:
            print(f"\n⏸️  No signal (waiting for 2σ outlier + overextension)")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_volatility_strategy())
