"""
Live Continuous Test for Volatility Mean Reversion Strategy

Monitors multiple instruments continuously and executes trades when 2-sigma signals appear.
Runs for a specified duration and tracks all results.
"""

import asyncio
import time
from datetime import datetime, timedelta
from iqclient import IQOptionAPI
from volatility_mean_reversion_strategy import (
    get_volatility_mean_reversion_signal,
    get_volatility_stats
)

# Test configuration
TEST_DURATION_MINUTES = 30
EXPIRY_MINUTES = 1
TRADE_AMOUNT = 1.0
CHECK_INTERVAL_SECONDS = 10  # Check every 10 seconds

# Test assets
TEST_ASSETS = [
    "EURUSD-OTC",
    "GBPUSD-OTC", 
    "USDJPY-OTC",
    "AUDUSD-OTC",
    "BTCUSD"
]

# Results tracking
trades_executed = []
total_pnl = 0.0
starting_balance = 0.0

async def execute_test_trade(api, asset, signal):
    """Execute a single test trade and wait for result."""
    global total_pnl
    
    try:
        print(f"\n🚀 Executing {signal} on {asset} (${TRADE_AMOUNT}, {EXPIRY_MINUTES}m expiry)")
        
        # Execute trade - USE DIGITAL OPTIONS
        success, order_id = await api.execute_digital_option_trade(
            asset, 
            TRADE_AMOUNT, 
            signal, 
            expiry=EXPIRY_MINUTES
        )
        
        if not success:
            print(f"❌ Trade failed: {order_id}")
            trades_executed.append({
                'time': datetime.now(),
                'asset': asset,
                'signal': signal,
                'result': 'FAILED',
                'pnl': 0.0
            })
            return
        
        print(f"✅ Trade placed - Order ID: {order_id}")
        print(f"⏳ Waiting {EXPIRY_MINUTES} minute(s) for result...")
        
        # Wait for trade to complete
        pnl_ok, pnl = await api.get_trade_outcome(
            order_id, 
            expiry=EXPIRY_MINUTES,
            asset_name=asset,
            direction=signal
        )
        
        # Record result
        result = "WIN" if (pnl_ok and pnl > 0) else "LOSS"
        total_pnl += pnl if pnl else 0.0
        
        trades_executed.append({
            'time': datetime.now(),
            'asset': asset,
            'signal': signal,
            'result': result,
            'pnl': pnl if pnl else 0.0
        })
        
        balance = api.get_current_account_balance()
        
        if result == "WIN":
            print(f"✅ WIN! Profit: ${pnl:.2f} | Total PnL: ${total_pnl:.2f} | Balance: ${balance:.2f}")
        else:
            print(f"❌ LOSS! Loss: ${pnl:.2f} | Total PnL: ${total_pnl:.2f} | Balance: ${balance:.2f}")
            
    except Exception as e:
        print(f"⚠️ Error executing trade: {e}")
        trades_executed.append({
            'time': datetime.now(),
            'asset': asset,
            'signal': signal,
            'result': 'ERROR',
            'pnl': 0.0
        })

async def run_live_volatility_test():
    """Run live volatility strategy test for specified duration."""
    global starting_balance, total_pnl
    
    print("=" * 80)
    print("LIVE VOLATILITY MEAN REVERSION TEST")
    print("=" * 80)
    print(f"\n⚙️ Configuration:")
    print(f"   Duration: {TEST_DURATION_MINUTES} minutes")
    print(f"   Expiry: {EXPIRY_MINUTES} minute(s)")
    print(f"   Trade Amount: ${TRADE_AMOUNT}")
    print(f"   Check Interval: {CHECK_INTERVAL_SECONDS} seconds")
    print(f"   Assets: {', '.join(TEST_ASSETS)}")
    
    print(f"\n📊 Strategy:")
    print(f"   - Detect 2-sigma candle body outliers")
    print(f"   - Confirm with engulfing pattern")
    print(f"   - Check overextension (BB, RSI, S/R)")
    print(f"   - Trade the reversion")
    
    # Initialize API
    print(f"\n🔌 Connecting to IQ Option...")
    api = IQOptionAPI()
    await api._connect()
    
    starting_balance = api.get_current_account_balance()
    print(f"✅ Connected! Starting Balance: ${starting_balance:.2f}")
    
    # Calculate end time
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=TEST_DURATION_MINUTES)
    
    print(f"\n⏰ Test Start: {start_time.strftime('%H:%M:%S')}")
    print(f"⏰ Test End:   {end_time.strftime('%H:%M:%S')}")
    print(f"\n{'='*80}")
    print("MONITORING FOR 2-SIGMA SIGNALS...")
    print("=" * 80)
    
    # Main test loop
    check_count = 0
    while datetime.now() < end_time:
        remaining = (end_time - datetime.now()).total_seconds()
        check_count += 1
        
        print(f"\n⏱️  Check #{check_count} | Time Remaining: {int(remaining/60)}m {int(remaining%60)}s")
        
        # Check each asset for signals
        for asset in TEST_ASSETS:
            try:
                # Fetch latest 1-minute candles
                candles = api.get_candle_history(asset, 300, 60)
                
                if not candles:
                    continue
                
                # Get volatility statistics
                stats = get_volatility_stats(candles, asset)
                
                # Get signal
                signal = get_volatility_mean_reversion_signal(candles, asset)
                
                # Display status
                if stats and (stats['is_outlier'] or signal):
                    print(f"\n📊 {asset}:")
                    print(f"   Sigma: {stats['sigma_distance']:.2f}σ | Outlier: {'✅' if stats['is_outlier'] else '❌'}")
                    print(f"   Signal: {signal or '-'}")
                
                # Execute trade if signal found
                if signal:
                    print(f"\n🎯 2-SIGMA SIGNAL FOUND: {asset} → {signal}")
                    await execute_test_trade(api, asset, signal)
                    
            except Exception as e:
                print(f"⚠️ Error checking {asset}: {e}")
        
        # Wait before next check
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
    
    # Test complete - show results
    print("\n" + "=" * 80)
    print("TEST COMPLETE - FINAL RESULTS")
    print("=" * 80)
    
    final_balance = api.get_current_account_balance()
    
    print(f"\n💰 BALANCE:")
    print(f"   Starting: ${starting_balance:.2f}")
    print(f"   Final:    ${final_balance:.2f}")
    print(f"   Change:   ${final_balance - starting_balance:.2f}")
    
    print(f"\n📊 TRADE SUMMARY:")
    print(f"   Total Trades: {len(trades_executed)}")
    
    if trades_executed:
        wins = sum(1 for t in trades_executed if t['result'] == 'WIN')
        losses = sum(1 for t in trades_executed if t['result'] == 'LOSS')
        win_rate = (wins / len(trades_executed) * 100) if trades_executed else 0
        
        print(f"   Wins:         {wins}")
        print(f"   Losses:       {losses}")
        print(f"   Win Rate:     {win_rate:.1f}%")
        print(f"   Total PnL:    ${total_pnl:.2f}")
        
        print(f"\n📋 TRADE LOG:")
        print(f"   {'Time':<10} {'Asset':<12} {'Signal':<6} {'Result':<8} {'PnL':<10}")
        print(f"   {'-'*60}")
        for trade in trades_executed:
            time_str = trade['time'].strftime('%H:%M:%S')
            pnl_str = f"${trade['pnl']:+.2f}"
            print(f"   {time_str:<10} {trade['asset']:<12} {trade['signal']:<6} {trade['result']:<8} {pnl_str:<10}")
    else:
        print(f"   ⚠️ No trades executed during test period")
        print(f"   This means no 2-sigma outliers occurred")
        print(f"   Try:")
        print(f"   - Running for longer duration")
        print(f"   - Testing during high volatility periods")
        print(f"   - Lowering SIGMA_THRESHOLD to 1.5 in strategy file")
    
    print("\n" + "=" * 80)
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"volatility_test_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("VOLATILITY MEAN REVERSION TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Duration: {TEST_DURATION_MINUTES} minutes\n")
        f.write(f"Expiry: {EXPIRY_MINUTES} minute(s)\n")
        f.write(f"Trade Amount: ${TRADE_AMOUNT}\n")
        f.write(f"Strategy: 2-Sigma Outlier + Mean Reversion\n\n")
        f.write(f"Starting Balance: ${starting_balance:.2f}\n")
        f.write(f"Final Balance: ${final_balance:.2f}\n")
        f.write(f"Change: ${final_balance - starting_balance:.2f}\n\n")
        f.write(f"Total Trades: {len(trades_executed)}\n")
        if trades_executed:
            f.write(f"Wins: {wins}\n")
            f.write(f"Losses: {losses}\n")
            f.write(f"Win Rate: {win_rate:.1f}%\n")
            f.write(f"Total PnL: ${total_pnl:.2f}\n\n")
            f.write("Trade Log:\n")
            for trade in trades_executed:
                f.write(f"{trade['time'].strftime('%H:%M:%S')} | {trade['asset']} | {trade['signal']} | {trade['result']} | ${trade['pnl']:+.2f}\n")
    
    print(f"📄 Results saved to: {filename}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_live_volatility_test())
