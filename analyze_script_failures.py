"""
Analyze Script.txt engulfing strategy failures
Find patterns in wins vs losses
"""
import pandas as pd
import numpy as np

# Load the training data
df = pd.read_csv('training_data.csv')

print("="*80)
print("SCRIPT.TXT ENGULFING STRATEGY - WIN/LOSS ANALYSIS")
print("="*80)

# Separate wins and losses
wins = df[df['outcome'] == 1]
losses = df[df['outcome'] == 0]

print(f"\nTotal Trades: {len(df)}")
print(f"Wins: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")

# Analyze by signal type
print("\n" + "="*80)
print("WIN RATE BY SIGNAL TYPE")
print("="*80)
for signal in ['CALL', 'PUT']:
    signal_df = df[df['signal'] == signal]
    signal_wins = signal_df[signal_df['outcome'] == 1]
    win_rate = len(signal_wins) / len(signal_df) * 100 if len(signal_df) > 0 else 0
    print(f"{signal:4} signals: {len(signal_df):4} | Win rate: {win_rate:5.1f}%")

# Analyze by asset
print("\n" + "="*80)
print("WIN RATE BY ASSET")
print("="*80)
for asset in df['asset'].unique():
    asset_df = df[df['asset'] == asset]
    asset_wins = asset_df[asset_df['outcome'] == 1]
    win_rate = len(asset_wins) / len(asset_df) * 100 if len(asset_df) > 0 else 0
    print(f"{asset:15} {len(asset_df):4} trades | Win rate: {win_rate:5.1f}%")

# Compare key indicators between wins and losses
print("\n" + "="*80)
print("INDICATOR COMPARISON: WINS vs LOSSES")
print("="*80)

indicators = ['adx', 'rsi', 'macd_hist', 'atr_pct', 'body_size_pct', 
              'dist_to_support', 'dist_to_resistance', 'bb_pos', 'wick_bias']

print(f"{'Indicator':20} {'Wins (avg)':>12} {'Losses (avg)':>12} {'Difference':>12}")
print("-"*80)

for ind in indicators:
    if ind in df.columns:
        win_avg = wins[ind].mean()
        loss_avg = losses[ind].mean()
        diff = win_avg - loss_avg
        print(f"{ind:20} {win_avg:12.4f} {loss_avg:12.4f} {diff:12.4f}")

# Analyze engulfing strength (body size ratio)
print("\n" + "="*80)
print("ENGULFING STRENGTH ANALYSIS")
print("="*80)

# Calculate body size for current and previous candles
df['body_curr'] = abs(df['close'] - df['open'])
df['body_prev'] = abs(df['close'].shift(1) - df['open'].shift(1))
df['engulfing_ratio'] = df['body_curr'] / df['body_prev']

wins_ratio = wins['engulfing_ratio'].mean()
losses_ratio = losses['engulfing_ratio'].mean()

print(f"Average engulfing ratio (wins):   {wins_ratio:.3f}")
print(f"Average engulfing ratio (losses): {losses_ratio:.3f}")
print(f"Difference: {wins_ratio - losses_ratio:.3f}")

# Analyze SMA alignment
print("\n" + "="*80)
print("SMA ALIGNMENT ANALYSIS")
print("="*80)

# Check how far price is from SMAs
for sma in ['sma_3', 'sma_7', 'sma_200']:
    if sma in df.columns:
        df[f'dist_{sma}'] = (df['close'] - df[sma]) / df[sma] * 100
        
        win_dist = wins[f'dist_{sma}'].mean()
        loss_dist = losses[f'dist_{sma}'].mean()
        
        print(f"Distance to {sma:7} - Wins: {win_dist:6.2f}% | Losses: {loss_dist:6.2f}% | Diff: {win_dist-loss_dist:6.2f}%")

# Analyze market conditions
print("\n" + "="*80)
print("MARKET REGIME ANALYSIS")
print("="*80)

if 'regime_trending' in df.columns and 'regime_ranging' in df.columns:
    # Trending markets
    trending = df[df['regime_trending'] == 1]
    trending_wins = trending[trending['outcome'] == 1]
    trending_wr = len(trending_wins) / len(trending) * 100 if len(trending) > 0 else 0
    
    # Ranging markets
    ranging = df[df['regime_ranging'] == 1]
    ranging_wins = ranging[ranging['outcome'] == 1]
    ranging_wr = len(ranging_wins) / len(ranging) * 100 if len(ranging) > 0 else 0
    
    print(f"Trending markets: {len(trending):4} trades | Win rate: {trending_wr:5.1f}%")
    print(f"Ranging markets:  {len(ranging):4} trades | Win rate: {ranging_wr:5.1f}%")

# Time-based analysis
print("\n" + "="*80)
print("TIME-BASED ANALYSIS")
print("="*80)

if 'hour' in df.columns:
    for hour_range, label in [
        (range(0, 8), "Asian (0-8)"),
        (range(8, 16), "London (8-16)"),
        (range(16, 24), "NY (16-24)")
    ]:
        session_df = df[df['hour'].isin(hour_range)]
        if len(session_df) > 0:
            session_wins = session_df[session_df['outcome'] == 1]
            session_wr = len(session_wins) / len(session_df) * 100
            print(f"{label:15} {len(session_df):4} trades | Win rate: {session_wr:5.1f}%")

# Key findings summary
print("\n" + "="*80)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*80)

# Find the biggest differences
print("\n1. BIGGEST INDICATOR DIFFERENCES (Wins vs Losses):")
diffs = {}
for ind in indicators:
    if ind in df.columns:
        win_avg = wins[ind].mean()
        loss_avg = losses[ind].mean()
        diffs[ind] = abs(win_avg - loss_avg)

top_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]
for ind, diff in top_diffs:
    win_avg = wins[ind].mean()
    loss_avg = losses[ind].mean()
    direction = "higher" if win_avg > loss_avg else "lower"
    print(f"   - {ind}: Wins have {direction} values (diff: {diff:.4f})")

print("\n2. POTENTIAL IMPROVEMENTS:")
print("   - Add filters based on indicators with biggest differences")
print("   - Focus on specific market sessions if one performs better")
print("   - Adjust engulfing ratio threshold if there's a clear pattern")
print("   - Consider combining with S/R levels for better entry points")

print("\n" + "="*80)
