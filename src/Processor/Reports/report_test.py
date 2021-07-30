import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
from tensorflow.python.keras.utils.np_utils import normalize

pd.options.mode.chained_assignment = None  # default="warn"

class Report_test():
    def __init__(self, x, y, pred, y_prove, x_prove, description, file, type, model) -> None:
        # l_name = [str(i) + "_" + str(j) for i, j in zip(model.keys(), model.values())]
        # name = "_".join(l_name)
        name = 'epochs_' + str(model['epochs'])
        self._path = "./src/Processor/Models/" + model["currency"] + model["time"] + "/" + model["table"] + "/analysis/csv/" + name + ".csv"

        # variavel X - inativa devido estar em scallar
        self.x_description, self.y_description = description
        df = x_prove
        df["y_prove"] = y_prove

        # cria df do y
        df_y = pd.DataFrame(pred, columns=self.y_description["columns"])
        df_y = df_y.T
        df_y.columns = ["y"]
        df_y = df_y.sort_values(by="y", ascending=False)

        df["y"] = np.nan
        df["pred"] = np.nan
        df["pred_percent"] = np.nan
        df["money_return"] = np.zeros(len(df))

        df["y"][-1:] = y[0]
        df["pred_percent"][-1:] = df_y["y"][:1]
        df["pred"][-1:] = df_y.index[:1]
        df = df.iloc[-1:, :]

        try:
            df_aux = pd.read_csv(self._path, index_col="date_time")
            df = df.append(df_aux)
        except:
            print("file not found")

        self.df = df
        self.df["pred"] = self.df["pred"].astype("int32")
        self.df["y"] = self.df["y"].astype("int32")
        self.df["right"] = self.df["pred"] == self.df["y"]
        self.df["pred_percent"] = pd.to_numeric(self.df["pred_percent"])

        # função de lucro sob o dataframe
        self.money_return()

        # toda alteração no dataframe deve ser feita antes de salvar
        df.to_csv(self._path)
  
    def print(self) -> None:
        print("___________________________________________________________________________________________________________________________________________________________________________________________________________________")
        print("\n")
        print(self.df.loc[:, ["Close", "high", "Low", "y", "high_bool__low_bool", "y_prove", "pred", "pred_percent", "money_return", "right"]])
        
        # filtra os acertos/erros a partir de uma referencia
        self.filter_percent([40, 50 ,60 ,70 ,80 ,90])

    def plot(self) -> None:
        pass
        
    def save(self) -> None:
        pass

    def filter_percent(self, percent: int) -> None:
        for i in percent:
            df = self.df[(self.df["pred_percent"] >= (i / 100) )]
            
            print("___________________________________________________________________________________________________________________________________________________________________________________________________________________")
            print("___________________________________________________________________________________________________________________________________________________________________________________________________________________")
            print("\tpercentual -> " + str(i) + "%")

            print("\n")
            pred = df["right"]
            print(pred.value_counts())
            print(pred.value_counts(normalize=True))

            print("\n")
            print(df.loc[:, ["right", "pred", "y"]].groupby(['y', 'pred']).count())

            try:
                self.print_money_return(i)
            except:
                print("\t...")

    def money_return(self) -> None:
        if (self.df["pred"][0] == -1):
            self.calc_money_return("high", "Low")
        elif (self.df["pred"][0] == 1):
            self.calc_money_return("Low", "high")
        else:
            self.df["money_return"][0] = 0

    def calc_money_return(self, stop: str, goal: str) -> None:
        if (self.df["y"][0] == self.df["pred"][0]):
            self.calc_percent_money_return(goal, 1, -1)
        elif (self.df["y"][0] == 0):
            i = 0
            while ((self.df["money_return"][0] == 0) & (i < len(self.df) - 1)):
                i = i + 1
                self.sub_calc_money_return(i, stop, goal)
        else:
            self.calc_percent_money_return(stop, -1, 1)
    
    def sub_calc_money_return(self, i: int, stop: str, goal: str) -> None:
        count = 0
        if ((self.df[goal][i] < self.df["Low"][0]) | (self.df[goal][i] > self.df["high"][0])):
            count = count + 1
            self.calc_percent_money_return(goal, 1, -1)
        if ((self.df[stop][i] < self.df["Low"][0]) | (self.df[stop][i] > self.df["high"][0])):
            count = count + 1
            self.calc_percent_money_return(stop, -1, 1)
        if (count == 2):
            self.df["money_return"][0] = None

    def calc_percent_money_return(self, x: str, a: int, b: int) -> None:
        money_return = (self.df["Close"][0] / self.df[x][0])
        money_return = money_return + 1 if money_return < 0 else money_return - 1
        money_return = money_return * b if money_return < 0 else money_return * a
        self.df["money_return"][0] = money_return
    
    def print_money_return(self, percent: int):

        # parametros da plataforma
        multiplicador = 500  # multiplcador de $
        percentual = 0.12 # percentual minimo de lucro ou prejuizo da plataforma
        percentual_max = 0.99 # percentual minimo de lucro ou prejuizo da plataforma
        investimento = 10 # reais
        taxa = 0.42 # relacioando a -> investimento = 10 
        
        df_money = self.df[self.df["pred_percent"] >= (percent / 100)]
        df_percent_range = df_money[((df_money["money_return"] > (percentual / multiplicador)) & (df_money["money_return"] < (percentual_max / multiplicador))) | (((df_money["money_return"] > -(percentual_max / multiplicador))& (df_money["money_return"]  < -(percentual / multiplicador))))]
        trade_in = df_percent_range[df_percent_range["pred"] != 0]["pred"].count() # numero de trades efetuados

        print("\n\tLucro/Prejuizo real ~= R$ : " + str(round(df_percent_range["money_return"].sum() * multiplicador * investimento - (trade_in * taxa), 2)))
        print("\tLucro/Prejuizo sem taxas ~= R$ : " + str(round((df_percent_range["money_return"].sum() * multiplicador * investimento), 2)))
        print("\tLucro/Prejuizo sem taxas e minimos ~= R$ : " + str(round((df_money["money_return"].sum() * multiplicador * investimento), 2)))
        
