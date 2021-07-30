import os
import sys
import time

import numpy as np
import pandas as pd
import ta
from pandas.core.frame import DataFrame

from Processor.Data_manager.indicators_analysis.generate_labels import Genlabels
from Processor.Data_manager.indicators_analysis.macd import Macd
from Processor.Data_manager.indicators_analysis.rsi import StochRsi
from Processor.Data_manager.indicators_analysis.dpo import Dpo
from Processor.Data_manager.indicators_analysis.coppock import Coppock
from Processor.Data_manager.indicators_analysis.poly_interpolation import PolyInter
from Processor.Data_manager.indicators_analysis.sar_parabolic import Parabolic_sar
from Processor.Data_manager.indicators_analysis.date_time import Date_time

class Data_x():
    def __init__(self, df: DataFrame, features: dict, mode: str) -> None:
        _path = "./src/Processor/Models/" + features["currency"] + features["time"] + "/" + features["table"]
        _path_data = _path + "/data.csv"
        _path_df = _path + "/df.csv"

        self.reduce = features['data']['reduce']
        size = len(df)

        try:
            if ((mode == "pred") | (mode == "die") | (mode == "gen")): 0/0
            # dataset pré-processado para testes
            self.x = pd.read_csv(_path_data)
            self.x = self.x.set_index("date_time")
            self.init_y = size % self.reduce

        except:
            # gera o dataframe processado
            # x_aux = self.adjust_columns(df, features['data']['predict']['columns'])
            x_aux = df.set_index("date_time")

            for i in range(size, 0, -self.reduce): # redução de dataset em multiplos dataframes
                if (i < self.reduce):
                    self.init_y = i
                    break

                self.x_temp = x_aux.iloc[(i-self.reduce):i, :]

                # self.x_temp = ta.add_all_ta_features(
                #     self.x_temp, "Open", "High", "Low", "Close", "Volume", fillna=True
                # )

                self.x_temp = ta.add_others_ta(
                    self.x_temp, "Close", fillna=True
                )

                self.x_temp = ta.add_trend_ta(
                    self.x_temp, "High", "Low", "Close", fillna=True
                )

                self.x_temp = ta.add_volatility_ta(
                    self.x_temp, "High", "Low", "Close", fillna=True
                )

                self.col_predict(['Close', 'High', 'Low'])
                self.col_labels(['High', 'Low'])
                self.col_macd(['High', 'Low'], False)
                self.col_rsi(['High', 'Low'], False)
                self.col_dpo(['High', 'Low'], False)
                self.col_coppock(['High', 'Low'], False)
                self.col_poly_interpolation(['High', 'Low'], False)
                self.col_parabolic_sar(['High', 'Low'], False, params={"af":0.02, "amax":0.2}, name='parabolic_sar')
                
                columns_cross = [
                                    ['High_bool', 'Low_bool'], 
                                    # ['close_bool', 'high_bool', 'low_bool'], 
                                    ['High_label_bool', 'Low_label_bool'], 
                                    # ['close_label_bool', 'high_label_bool', 'low_label_bool']
                                ]

                self.cross_bool_cols(columns_cross) 
                
                date = Date_time(self.x_temp)
                self.x_temp['weekday'] = date.Date()
                self.x_temp['hour'] = date.Time()

                if (i == size): self.x = self.x_temp.iloc[::-1]
                else: 
                    self.x = self.x.append(self.x_temp.iloc[::-1])

                if ((mode == "pred") | (mode == "die")): 
                    self.init_y = i - self.reduce
                    break
            
            self.x = self.x.iloc[::-1]

            if (mode != "pred"): 
                self.create_dir(_path)
                if (mode == "die"):
                    self.x.to_csv('./src/Processor/Notebooks/df.csv')
                else:
                    self.x.to_csv(_path_data)
                if (mode == 'gen'):
                    sys.exit()


    def adjust_columns(self, df, columns):
        # for i in df:
        #     if i not in columns:
        #        df = df.drop(columns=i)
        return df

    def convert_col_to_bool(self, col) -> None:
        name = col.name
        col = np.array(col)
        size = len(col)
        res = [0]

        for j in range(size - 1):
            if (j < size - 1):
                if (col[j] > col[j + 1]): res.append(0)
                else: res.append(1)
            else:
                res.append(None)
        self.x_temp[(name + "_bool")] = res

    def cross_bool_cols(self, cols) -> None:
        for i in cols:
            if (len(i) > 1): self.x_temp["__".join(i)] = [ 1 if j == len(i) else -1 if j == 0 else 0 for j in self.x_temp.loc[:, i].sum(axis=1)]

    def get_description(self) -> dict:
        return {"columns":  self.x.columns}

    def get_x(self) -> DataFrame:
        return self.x

    def create_dir(self, _path) -> None:
        if not os.path.exists(_path):
                    os.makedirs(_path)
                    os.makedirs(_path + "/.models")
                    os.makedirs(_path + "/analysis/csv")

    def get_init_y(self) -> int:
        return self.init_y

    # indicadores
    def col_predict(self, cols) -> None:
        for i in cols:
            self.convert_col_to_bool(self.x_temp[i])

    def col_labels(self, cols) -> None:
        for i in cols:
            self.x_temp[(i + '_label_bool')] = Genlabels(self.x_temp[i], {"window": 25, "polyorder": 3}).labels

    def col_macd(self, cols, bool_col) -> None:
        for i in cols:
            self.x_temp[(i + '_macd')] = Macd(self.x_temp[i], {'short_pd':12, 'long_pd':26, 'sig_pd':9}).values
            if (bool_col): self.convert_col_to_bool(self.x_temp[(i + '_macd')])

    def col_rsi(self, cols, bool_col) -> None:
        for i in cols:
            self.x_temp[(i + '_rsi')] = StochRsi(self.x_temp[i], {"period":14}).hist_values
            if (bool_col): self.convert_col_to_bool(self.x_temp[(i + '_rsi')])

    def col_dpo(self, cols, bool_col) -> None:
        for i in cols:
            self.x_temp[(i + '_dpo')] = Dpo(self.x_temp[i], {"period":4}).values
            if (bool_col): self.convert_col_to_bool(self.x_temp[(i + '_dpo')])

    def col_coppock(self, cols, bool_col) -> None:
        for i in cols:
            self.x_temp[(i + '_coppock')] = Coppock(self.x_temp[i], {"wma_pd":10, "roc_long":6, "roc_short":3}).values
            if (bool_col): self.convert_col_to_bool(self.x_temp[(i + '_coppock')])

    def col_poly_interpolation(self, cols, bool_col) -> None:
        for i in cols:
            self.x_temp[(i + '_poly_interpolation')] = PolyInter(self.x_temp[i], {"degree":4, "pd":20, "plot":False, "progress_bar":True} ).values
            if (bool_col): self.convert_col_to_bool(self.x_temp[(i + '_poly_interpolation')])

    def col_parabolic_sar(self, cols, bool_col, params={"af":0.02, "amax":0.2},name='parabolic_sar'):
        self.x_temp[name] = Parabolic_sar(self.x_temp.loc[:, cols], params, cols[0], cols[1]).values
        if (bool_col): self.convert_col_to_bool(self.x_temp[name])

    def col_date(self) -> None:
        self.x_temp['hour'] = pd.to_datetime(self.x_temp.index).hour
        self.x_temp['weekday'] = pd.to_datetime(self.x_temp.index).weekday
