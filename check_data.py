import pandas as pd

df = pd.read_csv('training_data.csv')
wr = df['outcome'].mean()
assets = df['asset'].nunique()

print(f'Total records: {len(df)}')
print(f'Win rate: {wr:.2%}')
print(f'Unique assets: {assets}')
print(f'\nAsset breakdown:')
print(df.groupby('asset')['outcome'].agg(['count', 'mean']))
