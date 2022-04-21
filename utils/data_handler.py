import pandas as pd
from constants import constants

class DataHandler:
    def dataframeToCSV(self, name, dataframe):
        path = name + '.csv'
        dataframe.to_csv(path)

    def CSVToDataframe(self, name):
        path = name + '.csv'
        return pd.read_csv(path,
                           header=0,
                           index_col='Timestamp',
                           parse_dates=True)

    def get_data(self):
        csvName = "".join(['data/', constants['symbol']])
        csv = self.CSVToDataframe(csvName)
        return csv

    def get_expert_data(self):
        csvName = "".join(['expert-data/', constants['symbol']])
        csv = self.CSVToDataframe(csvName)
        return csv