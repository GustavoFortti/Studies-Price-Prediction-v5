import pandas as pd

df = pd.read_csv('./data/EURUSD60_pred.csv').set_index('date_time')
df_n = pd.read_csv('./data/EURUSD60new.csv', sep='\t', names=['date_time', 'open', 'high', 'Low', 'Close', 'volume'])
df_n = df_n.loc[:, ['date_time', 'Close', 'high', 'Low']].iloc[-3:, :].set_index('date_time')

index = df_n.index
for i in index:
    df[df.index == i] = df_n[df_n.index == i]

df.to_csv('./data/EURUSD60_pred_.csv')
print(df)