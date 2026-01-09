"""
OTC Chaos Analysis - Finding Patterns in "Random" Markets

OTC pairs showed 49-50% with S/R bounce (almost random).
But "almost random" means there might be subtle patterns.

Let's test different hypotheses about OTC behavior.
"""

import pandas as pd

print("="*80)
print("OTC PAIRS - FINDING PATTERNS IN CHAOS")
print("="*80)

print("""
WHY OTC IS DIFFERENT:
1. Synthetic/simulated markets (not real order flow)
2. Broker-controlled pricing
3. May have different patterns than real markets
4. Available 24/7 (including weekends!)

WHAT WE KNOW:
- S/R bounce: 49-50% (almost random)
- Momentum: 0 signals (too strict)
- Engulfing: 46% (worse than random)

HYPOTHESES TO TEST:
1. OTC might be mean-reverting (opposite of trends)
2. OTC might follow time-based patterns (broker algorithms)
3. OTC might have micro-patterns (very short-term)
4. OTC might respond to volatility differently
5. OTC might have "fake" breakouts (traps)

Let's test each hypothesis...
""")

print("="*80)
print("HYPOTHESIS 1: OTC IS MEAN-REVERTING")
print("="*80)
print("""
Theory: OTC prices might snap back to mean more than real markets.

Test: Look for extreme RSI/BB positions and fade them.

Strategy:
- RSI > 70: Expect reversal DOWN (PUT)
- RSI < 30: Expect reversal UP (CALL)
- Price at BB extremes: Fade the move

Expected: 52-55% if OTC is mean-reverting
""")

print("="*80)
print("HYPOTHESIS 2: TIME-BASED PATTERNS")
print("="*80)
print("""
Theory: Broker algorithms might have predictable behavior at certain times.

Test: Analyze win rates by hour/session.

Strategy:
- Find hours with >52% win rate
- Only trade during those hours
- Avoid hours with <48% win rate

Expected: 51-53% if time patterns exist
""")

print("="*80)
print("HYPOTHESIS 3: MICRO-PATTERNS (1M TIMEFRAME)")
print("="*80)
print("""
Theory: OTC might have very short-term patterns that disappear on 5M.

Test: Use 1-minute candles with rapid signals.

Strategy:
- Look for 3-candle patterns
- Quick reversals
- Momentum bursts

Expected: 51-54% if micro-patterns exist
""")

print("="*80)
print("HYPOTHESIS 4: VOLATILITY EXPLOITATION")
print("="*80)
print("""
Theory: OTC volatility might be artificial and predictable.

Test: Trade based on ATR/volatility changes.

Strategy:
- High volatility: Expect continuation
- Low volatility: Expect breakout
- Volatility spike: Fade it

Expected: 51-53% if volatility is predictable
""")

print("="*80)
print("HYPOTHESIS 5: FAKE BREAKOUT TRAPS")
print("="*80)
print("""
Theory: OTC might create fake breakouts to trap traders.

Test: Fade breakouts instead of following them.

Strategy:
- Price breaks resistance: SHORT (PUT)
- Price breaks support: LONG (CALL)
- Opposite of normal trading!

Expected: 52-56% if OTC traps traders
""")

print("\n" + "="*80)
print("RECOMMENDED TESTING ORDER")
print("="*80)
print("""
1. FAKE BREAKOUT TRAPS (Easiest to test)
   - Reverse normal S/R logic
   - If works: Could be 52-56%

2. MEAN REVERSION (RSI extremes)
   - Simple RSI >70 / <30 signals
   - If works: Could be 52-55%

3. TIME-BASED PATTERNS
   - Analyze existing data by hour
   - Find profitable hours
   - If works: Could be 51-53%

4. VOLATILITY EXPLOITATION
   - ATR-based signals
   - If works: Could be 51-53%

5. MICRO-PATTERNS (1M)
   - Requires new data collection
   - Last resort if others fail

Which hypothesis should we test first?
""")

print("="*80)
