"""
Quick script to collect 5-minute data and compare with 1-minute results.
"""
import asyncio
from collect_data import run_collection_cycle

async def main():
    print("🚀 Collecting 5-minute candle data...")
    print("Expected improvements:")
    print("  - S/R levels more reliable")
    print("  - Technical patterns clearer")
    print("  - Win rate: 55-60% (vs 49% on 1M)")
    print()
    
    await run_collection_cycle()

if __name__ == "__main__":
    asyncio.run(main())
