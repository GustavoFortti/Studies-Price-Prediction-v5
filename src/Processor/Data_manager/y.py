import numpy as np
from pandas.core.frame import DataFrame

class Data_y():
    def __init__(self, df: DataFrame, features: dict) -> None:
        self.columns = features['data']['target']['columns']
        self.description = features['data']['target']['description']
        self.y = df.loc[:, self.columns]
        self.convert_col_to_bool()

    def convert_col_to_bool(self) -> None:
        for i in self.y:
            res = []
            size = len(self.y[i])
            col = np.array(self.y[i].reset_index().drop(columns='index'))
            for j in range(size):
                if (j < size - 1):
                    if (col[j] > col[j + 1]): res.append(0)
                    else: res.append(1)
                else:
                    res.append(None)
            self.y[i] = res
        self.y = [1 if i == len(self.columns) else -1 if i == 0 else 0 for i in self.y.sum(axis=1)]

    def get_description(self) -> dict:
        return {'columns': self.description}

    def get_y(self, init) -> np.array:
        return np.array(self.y[init:])