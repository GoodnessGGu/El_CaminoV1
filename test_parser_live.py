
import logging
from channel_signal_parser import parse_channel_signal

# Setup logging
logging.basicConfig(level=logging.INFO)

signal_text = """
🔔 NEW SIGNAL!

🎫 Trade: 🇬🇧 GBP/JPY 🇯🇵 (OTC)
⏳ Timer: 2 minutes
➡️ Entry: 12:53 PM
📈 Direction: SELL 🟥

↪️ Martingale Levels:
Level 1 → 12:55 PM
Level 2 → 12:57 PM
Level 3 → 12:59 PM
"""

print("Testing Parser with Screenshot Text...")
result = parse_channel_signal(signal_text)

if result:
    print("\n✅ SUCCESS: Signal Parsed!")
    print(f"Pair: {result['pair']}")
    print(f"Direction: {result['direction']}")
    print(f"Expiry: {result['expiry']}m")
    print(f"Time: {result['time']}")
else:
    print("\n❌ FAILED: Could not parse signal.")
