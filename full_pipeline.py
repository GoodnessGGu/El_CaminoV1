import asyncio
import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from settings import config
from collect_data import run_collection_cycle
from ml_utils import train_model as train_xgb
from ml_lstm import train_lstm
from iqclient import IQOptionAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Pipeline")

async def run_pipeline():
    print("\n" + "="*50)
    print("[*] STARTING FULL AI PIPELINE (No Telegram)")
    print("="*50 + "\n")
    
    # 1. Connect API
    print("[+] Connecting to IQ Option...")
    api = IQOptionAPI()
    await api._connect()
    
    if not api.check_connect():
        print("[-] Connect failed. Check credentials/internet.")
        return

    # 2. Collect Data
    print("\n[1/3] COLLECTING DATA...")
    await run_collection_cycle(api)
    
    # 3. Train Model
    print(f"\n[2/3] TRAINING MODEL: {config.model_type}...")
    if config.model_type == "LSTM":
        train_lstm()
    else:
        train_xgb()
        
    print("\n[+] Training Complete.")
    
    # 4. Backtest
    print("\n[3/3] RUNNING BACKTEST (EURGBP-OTC)...")
    # Import backtest here to ensure it uses the NEWLY trained model
    import backtest
    from importlib import reload
    reload(backtest) # Reload to pick up new model
    
    # Manually trigger backtest logic
    await backtest.run_backtest(api)
    
    print("\n" + "="*50)
    print("🏁 PIPELINE FINISHED")
    print("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        print("Stopped by user.")
