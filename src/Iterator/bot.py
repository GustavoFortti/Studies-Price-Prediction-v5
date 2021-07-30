import os
from time import sleep
import pandas as pd

class Interator():
    def __init__(self) -> None:
        df = pd.read_csv('./out/analisys.csv')
        df = df.iloc[-1:, :]
        sleep(3)

        self.macro = "xmacroplay -d 60 < ./src/Iterator/comands"
        self.trade_option = "high" if (df.direction.values[0] == 1) else "Low" if (df.direction.values[0] == -1) else None
        self.numbers_gain = [i for i in str(df.percent_gain.values[0])]
        self.numbers_loss = [i for i in str(df.percent_loss.values[0])]
    
    def run(self):
        if (self.trade_option):
            os.system(self.macro + "/trade/close_param")
            self.exec_nunbers(self.numbers_gain, "close_gain")
            self.exec_nunbers(self.numbers_loss, "close_loss")
            self.trade()

    def trade(self):
        os.system(self.macro + "/trade/" + self.trade_option)

    def exec_nunbers(self, numbers, option):
        sleep(1)
        os.system(self.macro + "/trade/" + option)
        os.system(self.macro + "/trade/clear")

        for i in numbers:
            sleep(1)
            os.system(self.macro + "/numbers/" + i)