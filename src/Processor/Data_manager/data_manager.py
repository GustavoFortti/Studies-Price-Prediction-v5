import sys
import time

import pandas as pd
import numpy as np

from keras.utils import to_categorical
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Processor.Data_manager.x import Data_x
from Processor.Data_manager.y import Data_y

class Data_manager():
    def __init__(self, file: str, mode: str, index: int, features: dict) -> None:
        start_time = time.time()
        # *********************************************
        # Alterações futuras:
        # desenvolver metodo para slice de dados unico, buscando o minimo de manipulação 
        # *********************************************]
        # file = file + '_pred' if (mode == 'pred') else file
        df = pd.read_csv('./data/' + file + '.csv', names=['date_time', "Open", "High", "Low", "Close", "Volume"], sep='\t')

        X = Data_x(df, features, mode)
        Y = Data_y(df, features)
        
        x = X.get_x()
        y = Y.get_y(X.get_init_y())

        # *********************************************
        # Desenvolver todo o processo novamente buscando entender se os dados estão na sequencia correta
        # tratar o modelo de teste de forma diferentes - variaveis prove*
        # validação
        # print(x.iloc[31:, :].head(15))
        # print(x.iloc[31:, :].head(15)['high_bool__low_bool'])
        # print(y[38:])
        # *********************************************

        if ((mode == 'die') | (mode == 'gen')): 
            print(x)
            print(x.columns)
            # sys.exit()

        if (mode != 'pred'):
            self.x_prove = x.iloc[-(index + features['timesteps'] + 1): -(1 + index), :]
        else:
            self.x_prove = x.iloc[-features['timesteps']:, :]

        print(self.x_prove)
        
        x, y = self.pre_shape_data(x, y, features['timesteps'], features['data']['reduce']) # divide o dataframe em bloco de 3d

        if (mode == 'train'): 
            x = x[:-1200] # retirando 2 meses de dados para teste
            y = y[:-1200]
            self.adjust_data(x, y, features)
        elif (mode == 'test'):
            self.x = x[-(2 + index):-(1 + index)] # pega apenas 1 bloco para fazer a predição dele
            self.y_prove = y[-(index + features['timesteps'] + 1):-(1 + index)] # cria um segundo y que sera levado diretamente ao relatorio para confirmar a resposta
            self.y = y[-(2 + index):-(1 + index)]
        else:
            self.x = x[-1:] # predição pega apenas o ultimo bloco

        self.x_description = X.get_description()
        self.y_description = Y.get_description()
        
        if ((mode == 'die') | (mode == 'gen')): 
            print("--- %s seconds data ---" % (time.time() - start_time))
            sys.exit()

        print(self.x_description)
        print(self.y_description)

    def pre_shape_data(self, x: DataFrame, y: np.array, timesteps: int, reduce: int) -> list:
        x_temp = []
        y_temp = []
        init = 31
        for i in range(0, len(x), reduce):
            x_aux, y_aux = self.shape_data(x.iloc[i + init:(i + reduce), :], y[i + init:(i + reduce)], timesteps)
            for i, j in zip(x_aux, y_aux): 
                x_temp.append(i) 
                y_temp.append(j) 

        return [np.array(x_temp), np.array(y_temp)]

    def shape_data(self, x: DataFrame, y: np.array, timesteps: int, init: int=31) -> list:
        scaler = StandardScaler() 
        x = scaler.fit_transform(x)

        reshaped = []
        for i in range(timesteps, x.shape[0] + 1):
            reshaped.append(x[i - timesteps:i])

        x = np.array(reshaped)
        y = np.array(y[timesteps-1:])

        return [x, y]

    def adjust_data(self, x: np.array, y: np.array, features: dict, split: float=0.3) -> None:
        # # count the number of each label
        # count_1 = np.count_nonzero(y)
        # count_0 = y.shape[0] - count_1
        # cut = min(count_0, count_1)

        # # save some data for testing
        # train_idx = int(cut * split)

        # # shuffle data
        # np.random.seed(42)

        # shuffle_index = np.random.permutation(x.shape[0])
   
        # x, y = x[shuffle_index], y[shuffle_index]

        # # find indexes of each label
        # idx_1 = np.argwhere(y == 1).flatten()
        # idx_0 = np.argwhere(y == 0).flatten()
        
        # # grab specified cut of each label put them together 
        # x_train = np.concatenate((x[idx_1[:train_idx]], x[idx_0[:train_idx]]), axis=0)
        # x_test = np.concatenate((x[idx_1[train_idx:cut]], x[idx_0[train_idx:cut]]), axis=0)
        # y_train = np.concatenate((y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
        # y_test = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

        # # shuffle again to mix labels
        # np.random.seed(7)
        # shuffle_train = np.random.permutation(x_train.shape[0])
        # shuffle_test = np.random.permutation(x_test.shape[0])
        # self.x_train, y_train = x_train[shuffle_train], y_train[shuffle_train]
        # self.x_test, y_test = x_test[shuffle_test], y_test[shuffle_test]

        categorical = features['data']['target']['categorical']

        self.x_train, self.x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
        self.y_train, self.y_test = to_categorical(y_train, categorical), to_categorical(y_test, categorical) 

    def get_train_test(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_y_prove(self):
        return self.y_prove

    def get_x_prove(self):
        return self.x_prove

    def get_descreption(self):
        return [self.x_description, self.y_description]