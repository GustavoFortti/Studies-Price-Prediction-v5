import pandas as pd
import numpy as np

class Date_time():
    def __init__(self, df) -> None:
        self.date = np.array(pd.to_datetime(df.index).weekday)
        self.time = np.array(pd.to_datetime(df.index).hour)

    def Date(self) -> pd.DataFrame:
        return self.date

    def Time(self) -> pd.DataFrame:
        return self.time
