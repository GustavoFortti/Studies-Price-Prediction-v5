import sys
import time

from Processor.model import Model

CONF_1 = {'type': 'LTSM', # parametros para o modelo 
    'type_x':'Data_x',    # tipo de entrada X da equação / trinit = high, Close e Low, high_label, close_label e low_label
    'type_y':'Data_y',    # tipo de entrada Y da equação / trinit = high, Close e Low, high_label, close_label e low_label
    'table': 'T3_C7_PRED',# T colunas e C versão
    'epochs': 30,         # epocas de treinamento
    'currency': 'EURUSD', # moeda utilizada
    'time': '60',         # Tempo de vela
    'timesteps': 8,       # quantidade de velas em um block X    
    'data': {
        "predict": {"columns": ['date_time', 'Close', 'High', 'Low'], "extend": [""]},
        "target": {"columns": ['High', 'Low'], "categorical": 3, "description": ["0", "1", "-1"]},
        "reduce": 1000
    }
}    

if __name__ == '__main__':
    args = sys.argv[1:]

    # modo da aplicação: teste, predição, treino, test data
    # index: escolhe bloco de dados para fazer testes
    start_time = time.time()
    model = Model(mode=args[1], index=int(args[0]), features=CONF_1)
    print("--- %s seconds ---" % (time.time() - start_time))

 
    