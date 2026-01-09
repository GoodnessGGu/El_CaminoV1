"""
Simple analysis of Script.txt patterns - focusing on what we can see
"""
import pandas as pd

df = pd.read_csv('training_data.csv')

wins = df[df['outcome'] == 1]
losses = df[df['outcome'] == 0]

print("="*80)
print("SCRIPT.TXT ENGULFING - SIMPLE PATTERN ANALYSIS")
print("="*80)

print(f"\nOverall: {len(wins)}/{len(df)} wins = {len(wins)/len(df)*100:.1f}%")

# Key insight: What if we add filters?
print("\n" + "="*80)
print("TESTING ADDITIONAL FILTERS TO IMPROVE WIN RATE")
print("="*80)

# Test 1: Add ADX filter (trending markets only)
for adx_threshold in [20, 25, 30]:
    filtered = df[df['adx'] > adx_threshold]
    if len(filtered) > 0:
        filtered_wins = filtered[filtered['outcome'] == 1]
        wr = len(filtered_wins) / len(filtered) * 100
        print(f"ADX > {adx_threshold:2}: {len(filtered):4} trades | Win rate: {wr:5.1f}%")

# Test 2: Add RSI filter (avoid extremes)
print("\nRSI Filters:")
for rsi_min, rsi_max in [(30, 70), (35, 65), (40, 60)]:
    filtered = df[(df['rsi'] > rsi_min) & (df['rsi'] < rsi_max)]
    if len(filtered) > 0:
        filtered_wins = filtered[filtered['outcome'] == 1]
        wr = len(filtered_wins) / len(filtered) * 100
        print(f"RSI {rsi_min}-{rsi_max}: {len(filtered):4} trades | Win rate: {wr:5.1f}%")

# Test 3: Combine filters
print("\nCombined Filters:")
for adx_thresh, rsi_min, rsi_max in [(25, 35, 65), (20, 40, 60), (30, 30, 70)]:
    filtered = df[(df['adx'] > adx_thresh) & (df['rsi'] > rsi_min) & (df['rsi'] < rsi_max)]
    if len(filtered) > 0:
        filtered_wins = filtered[filtered['outcome'] == 1]
        wr = len(filtered_wins) / len(filtered) * 100
        print(f"ADX>{adx_thresh} + RSI {rsi_min}-{rsi_max}: {len(filtered):4} trades | Win rate: {wr:5.1f}%")

# Test 4: Add S/R proximity filter (combine with S/R bounce!)
print("\nCombining with S/R Levels:")
for dist_threshold in [0.005, 0.003, 0.002, 0.001]:
    # Near support or resistance
    near_sr = df[(df['dist_to_support'] < dist_threshold) | (df['dist_to_resistance'] < dist_threshold)]
    if len(near_sr) > 0:
        near_sr_wins = near_sr[near_sr['outcome'] == 1]
        wr = len(near_sr_wins) / len(near_sr) * 100
        print(f"Near S/R (<{dist_threshold*100:.1f}%): {len(near_sr):4} trades | Win rate: {wr:5.1f}%")

# Test 5: Body size filter (stronger engulfing)
print("\nBody Size Filters:")
for body_threshold in [0.05, 0.08, 0.10, 0.12]:
    filtered = df[df['body_size_pct'] > body_threshold]
    if len(filtered) > 0:
        filtered_wins = filtered[filtered['outcome'] == 1]
        wr = len(filtered_wins) / len(filtered) * 100
        print(f"Body > {body_threshold*100:.0f}%: {len(filtered):4} trades | Win rate: {wr:5.1f}%")

# Best combination
print("\n" + "="*80)
print("BEST FILTER COMBINATION")
print("="*80)

# Try the most promising combination
best = df[
    (df['adx'] > 20) &  # Trending market
    (df['rsi'] > 35) & (df['rsi'] < 65) &  # Not extreme
    ((df['dist_to_support'] < 0.002) | (df['dist_to_resistance'] < 0.002))  # Near S/R
]

if len(best) > 0:
    best_wins = best[best['outcome'] == 1]
    best_wr = len(best_wins) / len(best) * 100
    print(f"\nADX>20 + RSI 35-65 + Near S/R:")
    print(f"  Trades: {len(best)}")
    print(f"  Win rate: {best_wr:.1f}%")
    print(f"  Improvement: {best_wr - 46.5:.1f}% vs baseline")
    
    if best_wr > 52:
        print("\n✅ THIS COMBINATION BEATS S/R BOUNCE!")
    elif best_wr > 50:
        print("\n⚠️ Profitable but still below S/R bounce (54%)")
    else:
        print("\n❌ Still not profitable")
else:
    print("No trades match this combination")

print("\n" + "="*80)
