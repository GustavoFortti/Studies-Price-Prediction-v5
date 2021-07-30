import sys

from Processor.Data_manager.data_manager import Data_manager
from Processor.Classfiers.LTSM.ltsm import LTSM_model
from Processor.Reports.report_test import Report_test
from Processor.Reports.report_pred import Report_pred

class Model():
    def __init__(self, mode: str, index: int, features: dict):
        self.file = features['currency'] + features['time']
        self.type = features['type_x'] + '_' + features['type_y']
        self.features = features

        
        data = Data_manager(self.file, mode, index, features)

        if (mode == 'train'):
            x_train, x_test, y_train, y_test = data.get_train_test()
            self.generate_model(x_train, x_test, y_train, y_test)
        elif (mode == 'test'):
            self.test_model(data.get_x(), data.get_y(), data.get_y_prove(), data.get_x_prove(), data.get_descreption())
        else:
            self.play_model(data.get_x(), data.get_x_prove(), features['data']['target']['description'])
        

    def generate_model(self, x_train, x_test, y_train, y_test) -> None:
        catalyst = LTSM_model(self.file, self.features)
        catalyst.create(x_train, x_test, y_train, y_test)
        catalyst.save()

    def test_model(self, x, y, y_prove, x_prove, description) -> None:
        catalyst = LTSM_model(self.file, self.features)
        pred = catalyst.predict(x)

        report = Report_test(x, y, pred, y_prove, x_prove, description, self.file, self.type, self.features)
        report.print()
        report.plot()
        report.save()

    def play_model(self, x, x_prove, y_description) -> None:
        catalyst = LTSM_model(self.file, self.features)
        pred = catalyst.predict(x)

        import pandas as pd
        import datetime

        df = pd.read_csv('./out/analisys.csv')

        df_y = pd.DataFrame(pred, columns=y_description)
        df_y = df_y.T
        df_y.columns = ["y"]
        df_y = df_y.sort_values(by="y", ascending=False)

        x_prove = x_prove.iloc[-1:, 1:4]
        High = abs((x_prove.High / x_prove.Close) - 1) * 500
        Low = abs((x_prove.Low / x_prove.Close) - 1) * 500
        
        gain = 0
        loss = 0

        if (int(df_y.index[:1][0]) == 1):
            gain = High.values[0]
            loss = Low.values[0]
        elif (int(df_y.index[:1][0]) == -1):
            gain = Low.values[0]
            loss = High.values[0]

        ax_date = datetime.datetime.now()
        date = None

        if (ax_date.minute >= 55):
            date = ax_date.replace(hour=ax_date.hour + 1)
        elif (ax_date.minute <= 5):
            date = ax_date

        if (date):
            df = df.append({"date_time": date.strftime("%Y-%m-%d %H:00"), "now": ax_date,"direction": df_y.index[:1][0],"percent_gain": int(gain * 100),"percent_loss": int(loss * 100), "percent_win": df_y["y"][:1][0]}, ignore_index=True)
            df.to_csv('./out/analisys.csv', index=False)
        print(df)