import pandas as pd

df = pd.read_csv('training_data.csv')

print('Win Rates by Asset (5M Timeframe):')
print('='*60)

results = []
for asset in sorted(df['asset'].unique()):
    asset_df = df[df['asset'] == asset]
    win_rate = asset_df['outcome'].mean() * 100
    count = len(asset_df)
    results.append((asset, win_rate, count))
    print(f'{asset:15} {win_rate:5.2f}% ({count:5} examples)')

print('='*60)
print(f'\nTotal: {len(df)} examples from {len(results)} assets')

# Show profitable pairs (>=52%)
profitable = [(a, w, c) for a, w, c in results if w >= 52.0]
print(f'\n✅ Profitable Pairs (>=52%): {len(profitable)}')
for asset, win_rate, count in profitable:
    print(f'  - {asset}: {win_rate:.2f}%')
