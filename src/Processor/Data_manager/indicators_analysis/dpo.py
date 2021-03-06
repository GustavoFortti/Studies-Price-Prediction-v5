import numpy as np
from pandas import DataFrame

class Dpo(object):
    def __init__(self, data: DataFrame, params: dict={'period':10}):
        self.period = params['period']
        self.data = None

        self.values = self.calc_dpo(np.array(data))

    def calc_dpo(self, data):
        dpo = [0] * (self.period - 1)

        for idx in range(self.period, len(data) + 1):
            self.data = data[idx - self.period:idx]
            sma = np.average(self.data)
            dpo.append(data[idx - int(self.period/2)] - sma)

        return np.array(dpo)

    def update_dpo(self, value):
        # update data for calculations
        self.data.append(value)
        self.data.pop(0)

        sma = np.average(self.data)

        return self.data[-(int(self.period/2))] - sma
        
    def get_values(self):
        return self.values