import os

from requests import get
from time import sleep
from datetime import datetime

while (True):
    print(datetime.now().strftime("%S"))
    if (datetime.now().strftime("%S") in ["01", "02", "03"]):
        while (True):
            second = datetime.now().strftime("%S")
            print("Current Time =", second)
            with open('./src/Extractor/currency.txt', 'a') as f:
                res = get('https://economia.awesomeapi.com.br/last/EUR-USD')
                if (res.status_code == 200):
                    print('write request')
                    data = res.json()['EURUSD']['bid']
                    print(data)
                    f.write(data + '\n')
                else:
                    print('error')
                f.Close()
            sleep(30)
            os.system('clear')
    sleep(1)
    os.system('clear')