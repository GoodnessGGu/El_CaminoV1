from iqclient import IQOptionAPI
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import asyncio

async def debug_history():
    api = IQOptionAPI()
    logger.info("Connecting...")
    await api._connect()
    
    # Wait for connection
    await asyncio.sleep(3)
    
    logger.info("Fetching History...")
    # Fetch recently closed positions
    # Need to check if get_position_history_by_page is async?
    # View file checks: It calls _send_position_query which uses send_message (sync?)
    # But it returns list immediately? 
    # Check iqclient.py Step 57. get_position_history_by_page delegates to account_manager.
    # Check account_manager in Step 54. 
    # It constructs msg and calls _send_position_query.
    # _send_position_query constructs a Future-like wait?
    # Let's assume it's synchronous-blocking style based on `markets` view.
    # But let's check if it needs await. `msg` sending is usually sync.
    # The return value is the list.
    
    # Actually, iqclient.py methods might not be async, but _connect IS async.
    # So `api.get_position_history_by_page` is likely sync.
    
    history = api.account_manager.get_position_history_by_page(["turbo-option", "binary-option"], limit=5, offset=0)
    
    if history:
        logger.info(f"✅ Found {len(history)} items.")
        item = history[0]
        logger.info("--- RAW ITEM STRUCTURE ---")
        for k, v in item.items():
            logger.info(f"{k}: {v}")
        logger.info("--------------------------")
    else:
        logger.warning("❌ No history found.")

if __name__ == "__main__":
    asyncio.run(debug_history())
