from iqclient import IQOptionAPI
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

def test_sentiment():
    email = os.getenv("IQ_EMAIL")
    password = os.getenv("IQ_PASSWORD")
    
    api = IQOptionAPI(email, password)
    # Sync connect wrapper
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connected = loop.run_until_complete(api.ensure_connect())
    
    if not connected:
        print("Failed to connect")
        return

    asset = "EURUSD" # Try standard pair
    print(f"Fetching mood for {asset}...")
    
    # get_traders_mood is synchronous (blocking wait)
    mood = api.market_manager.get_traders_mood(asset)
    print(f"Sentiment for {asset}: {mood}")
    
    # Try OTC
    asset_otc = "EURUSD-OTC"
    try:
        mood_otc = api.market_manager.get_traders_mood(asset_otc)
        print(f"Sentiment for {asset_otc}: {mood_otc}")
    except Exception as e:
        print(f"OTC mood failed: {e}")
        
    api.websocket.close()

if __name__ == "__main__":
    test_sentiment()
