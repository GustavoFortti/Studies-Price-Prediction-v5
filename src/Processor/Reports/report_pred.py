import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns

pd.options.mode.chained_assignment = None  # default="warn"

class Report_pred():
    def __init__(self, x) -> None:
        pass

    def y(self, pred):
        cols = ['0', '1', "-1"]
        df_y = pd.DataFrame(pred, columns=cols)
        df_y = df_y.T
        df_y.columns = ["y"]
        df_y = df_y.sort_values(by="y", ascending=False)
        direction = df_y.index[:1]
