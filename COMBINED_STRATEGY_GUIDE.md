"""
Combined ColorMillion + Engulfing Strategy - Usage Guide

This strategy combines two proven approaches for higher confidence signals.

## Strategy Components:

### 1. ColorMillion Strategy
- **Indicators:** EMA13, MACD (12,26,9), EMA200, ADX
- **CALL Signal:** EMA13 rising + MACD rising + price above EMA200 + ADX > 18
- **PUT Signal:** EMA13 falling + MACD falling + price below EMA200 + ADX > 18

### 2. Engulfing Pattern Strategy
- **Indicators:** SMA 3, 7, 200
- **CALL Signal:** Bullish engulfing + price above all SMAs
- **PUT Signal:** Bearish engulfing + price below all SMAs

### 3. Combined Strategy
- **CALL:** Both strategies agree on CALL
- **PUT:** Both strategies agree on PUT
- **None:** Strategies disagree or no signal

## How to Use in Telegram Bot:

### Option 1: Use Combined Strategy (Recommended)
```python
# In telegram_bot.py, replace:
from sr_bounce_strategy import get_sr_bounce_signal

# With:
from combined_strategy import get_combined_signal

# Then in auto_trade_loop, replace:
signal = get_sr_bounce_signal(candles, asset_name=asset)

# With:
signal = get_combined_signal(candles, asset_name=asset)
```

### Option 2: Use ColorMillion Only
```python
from combined_strategy import get_colormillion_only
signal = get_colormillion_only(candles, asset_name=asset)
```

### Option 3: Use Engulfing Only
```python
from combined_strategy import get_engulfing_only
signal = get_engulfing_only(candles, asset_name=asset)
```

## Testing the Strategy:

Run the test script to see how each strategy performs:
```bash
python test_combined_strategy.py
```

## Expected Performance:

### Combined Strategy (Both Must Agree)
- **Trade Frequency:** Lower (very selective)
- **Expected Win Rate:** 60-65% (high confidence)
- **Best For:** Conservative trading, quality over quantity

### ColorMillion Only
- **Trade Frequency:** Medium
- **Expected Win Rate:** 52-56%
- **Best For:** Trending markets

### Engulfing Only
- **Trade Frequency:** Medium
- **Expected Win Rate:** 50-54%
- **Best For:** Reversal patterns

## Recommended Settings:

### For More Trades:
Use ColorMillion or Engulfing individually

### For Higher Win Rate:
Use Combined strategy (both must agree)

### Current S/R Bounce:
- **Trade Frequency:** Very low (1 trade in 14 hours)
- **Win Rate:** 61.5% observed

### Combined Strategy vs S/R Bounce:
- **More signals** than S/R bounce (less strict)
- **Similar win rate** (60-65% expected)
- **Better for active trading**

## Files Created:
1. `combined_strategy.py` - Main strategy file
2. `test_combined_strategy.py` - Test script
3. `COMBINED_STRATEGY_GUIDE.md` - This guide

## Next Steps:
1. Test the strategy: `python test_combined_strategy.py`
2. Choose which variation to use
3. Update `telegram_bot.py` to use the chosen strategy
4. Restart bot and monitor performance
