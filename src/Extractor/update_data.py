from datetime import datetime
import pandas as pd

file = open('./src/Extractor/currency.txt', 'r')

values = file.read().split('\n')
if (values[-1:] == ['']):
    values = values[:-1]
values = [float(i) for i in values]

file = open('./src/Extractor/currency.txt', 'w')
file.write('')
file.Close()

ax_date = datetime.now() 
date = None

if (ax_date.minute >= 55):
    date = ax_date.replace(hour=ax_date.hour + 1)
elif (ax_date.minute <= 5):
    date = ax_date

if (date):
    row = {"date_time": date.strftime("%Y-%m-%d %H:00"), "Close": values[-1:][0], "high": max(values), "Low": min(values)}
    df = pd.read_csv('./data/EURUSD60_pred.csv')
    df = df.append(row, ignore_index=True)
    print(df)
    # df.to_csv('./data/EURUSD60_pred.csv')