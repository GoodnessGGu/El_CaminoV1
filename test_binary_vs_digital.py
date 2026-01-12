"""
Binary vs Digital Options Test

Tests both binary and digital option execution on non-OTC pairs.
Compares execution success, confirmation speed, and payout differences.
"""

import asyncio
import time
from datetime import datetime
from iqclient import IQOptionAPI

# Test configuration
TRADE_AMOUNT = 1.0
EXPIRY_MINUTES = 1

# Non-OTC pairs to test
TEST_PAIRS = [
    "EURUSD",      # Major forex
    "GBPUSD",      # Major forex
    "USDJPY",      # Major forex
    "BTCUSD",      # Crypto
    "ETHUSD",      # Crypto
    "XAUUSD",      # Gold
]

# Test results storage
test_results = []


async def test_digital_option(api, asset, direction):
    """Test digital option execution."""
    print(f"\n{'='*60}")
    print(f"📊 DIGITAL OPTION TEST: {asset} {direction}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Execute digital option
        success, result = await api.execute_digital_option_trade(
            asset,
            TRADE_AMOUNT,
            direction,
            expiry=EXPIRY_MINUTES
        )
        
        execution_time = time.time() - start_time
        
        if success:
            order_id = result
            print(f"✅ Digital option placed successfully")
            print(f"   Order ID: {order_id}")
            print(f"   Execution Time: {execution_time:.2f}s")
            
            # Wait for result
            print(f"⏳ Waiting {EXPIRY_MINUTES} minute(s) for outcome...")
            
            pnl_ok, pnl = await api.get_trade_outcome(
                order_id,
                expiry=EXPIRY_MINUTES,
                asset_name=asset,
                direction=direction
            )
            
            result_status = "WIN" if (pnl_ok and pnl > 0) else "LOSS"
            
            print(f"\n{'✅' if result_status == 'WIN' else '❌'} Result: {result_status}")
            print(f"   PnL: ${pnl:.2f}")
            
            return {
                'asset': asset,
                'type': 'DIGITAL',
                'direction': direction,
                'success': True,
                'execution_time': execution_time,
                'order_id': order_id,
                'result': result_status,
                'pnl': pnl
            }
        else:
            print(f"❌ Digital option failed: {result}")
            print(f"   Execution Time: {execution_time:.2f}s")
            
            return {
                'asset': asset,
                'type': 'DIGITAL',
                'direction': direction,
                'success': False,
                'execution_time': execution_time,
                'error': str(result),
                'result': 'FAILED',
                'pnl': 0.0
            }
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Exception: {e}")
        
        return {
            'asset': asset,
            'type': 'DIGITAL',
            'direction': direction,
            'success': False,
            'execution_time': execution_time,
            'error': str(e),
            'result': 'ERROR',
            'pnl': 0.0
        }


async def test_binary_option(api, asset, direction):
    """Test binary/turbo option execution."""
    print(f"\n{'='*60}")
    print(f"📊 BINARY OPTION TEST: {asset} {direction}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Execute binary option
        success, result = await api.execute_binary_option_trade(
            asset,
            TRADE_AMOUNT,
            direction,
            expiry=EXPIRY_MINUTES
        )
        
        execution_time = time.time() - start_time
        
        if success:
            order_id = result
            print(f"✅ Binary option placed successfully")
            print(f"   Order ID: {order_id}")
            print(f"   Execution Time: {execution_time:.2f}s")
            
            # Wait for result
            print(f"⏳ Waiting {EXPIRY_MINUTES} minute(s) for outcome...")
            
            pnl_ok, pnl = await api.get_binary_trade_outcome(
                order_id,
                expiry=EXPIRY_MINUTES,
                asset_name=asset,
                direction=direction
            )
            
            result_status = "WIN" if (pnl_ok and pnl > 0) else "LOSS"
            
            print(f"\n{'✅' if result_status == 'WIN' else '❌'} Result: {result_status}")
            print(f"   PnL: ${pnl:.2f}")
            
            return {
                'asset': asset,
                'type': 'BINARY',
                'direction': direction,
                'success': True,
                'execution_time': execution_time,
                'order_id': order_id,
                'result': result_status,
                'pnl': pnl
            }
        else:
            print(f"❌ Binary option failed: {result}")
            print(f"   Execution Time: {execution_time:.2f}s")
            
            return {
                'asset': asset,
                'type': 'BINARY',
                'direction': direction,
                'success': False,
                'execution_time': execution_time,
                'error': str(result),
                'result': 'FAILED',
                'pnl': 0.0
            }
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Exception: {e}")
        
        return {
            'asset': asset,
            'type': 'BINARY',
            'direction': direction,
            'success': False,
            'execution_time': execution_time,
            'error': str(e),
            'result': 'ERROR',
            'pnl': 0.0
        }


async def run_comparison_test():
    """Run comprehensive binary vs digital comparison test."""
    
    print("=" * 80)
    print("BINARY vs DIGITAL OPTIONS COMPARISON TEST")
    print("=" * 80)
    print(f"\n⚙️ Configuration:")
    print(f"   Trade Amount: ${TRADE_AMOUNT}")
    print(f"   Expiry: {EXPIRY_MINUTES} minute(s)")
    print(f"   Test Pairs: {', '.join(TEST_PAIRS)}")
    
    # Initialize API
    print(f"\n🔌 Connecting to IQ Option...")
    api = IQOptionAPI()
    await api._connect()
    
    starting_balance = api.get_current_account_balance()
    print(f"✅ Connected! Starting Balance: ${starting_balance:.2f}")
    
    print(f"\n{'='*80}")
    print("STARTING TESTS")
    print("=" * 80)
    
    # Test each pair with both option types
    for asset in TEST_PAIRS:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING: {asset}")
        print("#" * 80)
        
        # Determine direction (alternate between CALL and PUT)
        direction = "call" if TEST_PAIRS.index(asset) % 2 == 0 else "put"
        
        # Test Digital Option
        digital_result = await test_digital_option(api, asset, direction)
        test_results.append(digital_result)
        
        # Wait a bit between tests
        print("\n⏸️  Waiting 5 seconds before next test...")
        await asyncio.sleep(5)
        
        # Test Binary Option
        binary_result = await test_binary_option(api, asset, direction)
        test_results.append(binary_result)
        
        # Wait before next asset
        if asset != TEST_PAIRS[-1]:
            print("\n⏸️  Waiting 10 seconds before next asset...")
            await asyncio.sleep(10)
    
    # Final Results
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS - COMPARISON SUMMARY")
    print("=" * 80)
    
    final_balance = api.get_current_account_balance()
    balance_change = final_balance - starting_balance
    
    print(f"\n💰 BALANCE:")
    print(f"   Starting: ${starting_balance:.2f}")
    print(f"   Final:    ${final_balance:.2f}")
    print(f"   Change:   ${balance_change:+.2f}")
    
    # Analyze results by type
    digital_results = [r for r in test_results if r['type'] == 'DIGITAL']
    binary_results = [r for r in test_results if r['type'] == 'BINARY']
    
    print(f"\n📊 DIGITAL OPTIONS:")
    digital_success = sum(1 for r in digital_results if r['success'])
    digital_wins = sum(1 for r in digital_results if r['result'] == 'WIN')
    digital_total_pnl = sum(r['pnl'] for r in digital_results)
    avg_digital_time = sum(r['execution_time'] for r in digital_results) / len(digital_results) if digital_results else 0
    
    print(f"   Total Attempts:    {len(digital_results)}")
    print(f"   Successful Exec:   {digital_success} ({digital_success/len(digital_results)*100:.1f}%)")
    print(f"   Wins:              {digital_wins}")
    print(f"   Total PnL:         ${digital_total_pnl:+.2f}")
    print(f"   Avg Exec Time:     {avg_digital_time:.2f}s")
    
    print(f"\n📊 BINARY OPTIONS:")
    binary_success = sum(1 for r in binary_results if r['success'])
    binary_wins = sum(1 for r in binary_results if r['result'] == 'WIN')
    binary_total_pnl = sum(r['pnl'] for r in binary_results)
    avg_binary_time = sum(r['execution_time'] for r in binary_results) / len(binary_results) if binary_results else 0
    
    print(f"   Total Attempts:    {len(binary_results)}")
    print(f"   Successful Exec:   {binary_success} ({binary_success/len(binary_results)*100:.1f}%)")
    print(f"   Wins:              {binary_wins}")
    print(f"   Total PnL:         ${binary_total_pnl:+.2f}")
    print(f"   Avg Exec Time:     {avg_binary_time:.2f}s")
    
    # Detailed results table
    print(f"\n📋 DETAILED RESULTS:")
    print(f"   {'Asset':<12} {'Type':<8} {'Dir':<6} {'Status':<10} {'Result':<8} {'PnL':<10} {'Time':<8}")
    print(f"   {'-'*70}")
    
    for result in test_results:
        status = '✅ Success' if result['success'] else '❌ Failed'
        pnl_str = f"${result['pnl']:+.2f}" if result['success'] else 'N/A'
        time_str = f"{result['execution_time']:.2f}s"
        
        print(f"   {result['asset']:<12} {result['type']:<8} {result['direction'].upper():<6} "
              f"{status:<10} {result['result']:<8} {pnl_str:<10} {time_str:<8}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if digital_success > binary_success:
        print(f"   ✅ Digital options have higher success rate ({digital_success}/{len(digital_results)} vs {binary_success}/{len(binary_results)})")
        print(f"   → Recommend using DIGITAL as preferred type")
    elif binary_success > digital_success:
        print(f"   ✅ Binary options have higher success rate ({binary_success}/{len(binary_results)} vs {digital_success}/{len(digital_results)})")
        print(f"   → Recommend using BINARY as preferred type")
    else:
        print(f"   ⚖️  Both types have equal success rate")
        print(f"   → Keep AUTO mode for automatic fallback")
    
    if avg_digital_time < avg_binary_time:
        print(f"   ⚡ Digital options are faster ({avg_digital_time:.2f}s vs {avg_binary_time:.2f}s)")
    else:
        print(f"   ⚡ Binary options are faster ({avg_binary_time:.2f}s vs {avg_digital_time:.2f}s)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"binary_vs_digital_test_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("BINARY vs DIGITAL OPTIONS TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trade Amount: ${TRADE_AMOUNT}\n")
        f.write(f"Expiry: {EXPIRY_MINUTES} minute(s)\n\n")
        
        f.write(f"Starting Balance: ${starting_balance:.2f}\n")
        f.write(f"Final Balance: ${final_balance:.2f}\n")
        f.write(f"Change: ${balance_change:+.2f}\n\n")
        
        f.write("DIGITAL OPTIONS:\n")
        f.write(f"  Attempts: {len(digital_results)}\n")
        f.write(f"  Success: {digital_success} ({digital_success/len(digital_results)*100:.1f}%)\n")
        f.write(f"  Wins: {digital_wins}\n")
        f.write(f"  PnL: ${digital_total_pnl:+.2f}\n")
        f.write(f"  Avg Time: {avg_digital_time:.2f}s\n\n")
        
        f.write("BINARY OPTIONS:\n")
        f.write(f"  Attempts: {len(binary_results)}\n")
        f.write(f"  Success: {binary_success} ({binary_success/len(binary_results)*100:.1f}%)\n")
        f.write(f"  Wins: {binary_wins}\n")
        f.write(f"  PnL: ${binary_total_pnl:+.2f}\n")
        f.write(f"  Avg Time: {avg_binary_time:.2f}s\n\n")
        
        f.write("DETAILED RESULTS:\n")
        for result in test_results:
            f.write(f"{result['asset']} | {result['type']} | {result['direction'].upper()} | "
                   f"{'Success' if result['success'] else 'Failed'} | {result['result']} | "
                   f"${result['pnl']:+.2f} | {result['execution_time']:.2f}s\n")
    
    print(f"\n📄 Results saved to: {filename}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_comparison_test())
