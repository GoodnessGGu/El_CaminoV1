import pandas as pd
import os

files = ['backtest_EURUSD-OTC_60s.xlsx', 'backtest_GBPUSD-OTC_60s.xlsx']

for f in files:
    try:
        if not os.path.exists(f): 
            print(f"File not found: {f}")
            continue
            
        df = pd.read_excel(f)
        total = len(df)
        wins = len(df[df['win'] == True])
        wr = (wins / total * 100) if total > 0 else 0
        pnl = df['pnl'].sum()
        
        print(f"\n" + "-"*30)
        print(f"RESULTS: {f}")
        print(f"-"*30)
        print(f"Total Trades: {total}")
        print(f"Wins:         {wins}")
        print(f"Win Rate:     {wr:.2f}%")
        print(f"PnL:          ${pnl:.2f}")
        print(f"-"*30)
    except Exception as e:
        print(f"Error reading {f}: {e}")

