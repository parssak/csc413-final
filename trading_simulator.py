from trading_environment import TradingEnvironment
from ml.tdqn import TDQN
from utils.data_handler import DataHandler
from utils.performance_estimator import PerformanceEstimator
from utils.utils import generateTrainDataFromDF
from utils.indicators import applyIndicators
from constants import constants
from utils.documentor import Documentor
from matplotlib import pyplot as plt
import pandas as pd


class TradingSimulator:
    def simulateNewStrategy(self):
        tradingStrategy = TDQN()
        data = DataHandler().get_data()
        # data = applyIndicators(data)
        data = data[constants['dates']['start']:constants['dates']['end']]
        splitPct = constants['dates']['splitPct']
        trainData = data[:int(len(data) * splitPct)]
        testData = data[int(len(data) * splitPct):]
        splitDate = testData.index[0]
        print(trainData.shape)
        td = generateTrainDataFromDF(trainData)
        # td = applyIndicators(td)
        d = Documentor.instance()
        d.write("dates", {
            "start": constants['dates']['start'],
            "end": constants['dates']['end'],
            "split": str(splitDate),
            "splitPct": constants['dates']['splitPct'],
        })

        trainingEnvironment = TradingEnvironment(td)

        print("Training...")
        tradingStrategy.training(trainingEnvironment, testData)

        testingEnvironment = TradingEnvironment(testData)

        # trainingEnvironment.data = trainData

        print("Testing...")
        tradingStrategy.testing(trainingEnvironment, testingEnvironment, True)

        print("Storing Results...")
        self.storeResults(trainingEnvironment, testingEnvironment)

        d.save()

    def storeResults(self, trainingEnv, testingEnv):
        # Artificial trick to assert the continuity of the Money curve
        # get last money value from training env
        
        ratio = trainingEnv.data['Money'].tolist()[-1]/testingEnv.data['Money'][0]
        testingEnv.data['Money'] = ratio * testingEnv.data['Money']

        # Concatenation of the training and testing trading dataframes
        dataframes = [trainingEnv.data, testingEnv.data]
        data = pd.concat(dataframes)

        indices = data.index
        data = data[['Volume',
                     'Action',
                     'Money',
                     'Returns',
                     'Close',
                    #  'Holdings',
                     #  'Cash'
                     ]]

        data.rename(columns={
            'Volume': 'v',
            'Action': 'a',
            'Money': 'm',
            'Returns': 'r',
            'Close': 'c'
            # 'Holdings': 'h',
            # 'Cash': 'c'
            # 'Position': 'p'
        }, inplace=True)
        # data.rename(columns={'Timestamp': 't'}, inplace=True)

        # @parssa TODO: Finish compressing the saved results

        records = data.to_dict(orient="records")

        # add the indices to the records
        for i, record in enumerate(records):
            record["Timestamp"] = str(indices[i])

        # Performance Estimations
        trainingPerformance = PerformanceEstimator(
            trainingEnv.data).computePerformance()

        testingPerformance = PerformanceEstimator(
            testingEnv.data).computePerformance()

        d = Documentor.instance()
        d.write("data", records)
        d.write("performance", {
            "training": trainingPerformance,
            "testing": testingPerformance
        })
