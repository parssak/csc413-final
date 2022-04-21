import numpy as np
import math
from constants import constants
import json
import pandas as pd


def getNormalizedCoefficients(env):
    data = env.data
    closePrices = data['Close'].tolist()
    lowPrices = data['Low'].tolist()
    highPrices = data['High'].tolist()
    volumes = data['Volume'].tolist()
    # Retrieve the coefficients required for the normalization
    coefficients = []
    margin = 1
    # 1. Close price => returns (absolute) => maximum value (absolute)
    returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1])
               for i in range(1, len(closePrices))]
    coeffs = (0, np.max(returns)*margin)
    coefficients.append(coeffs)
    # 2. Low/High prices => Delta prices => maximum value
    deltaPrice = [abs(highPrices[i]-lowPrices[i])
                  for i in range(len(lowPrices))]
    coeffs = (0, np.max(deltaPrice)*margin)
    coefficients.append(coeffs)
    # 3. Close/Low/High prices => Close price position => no normalization required
    coeffs = (0, 1)
    coefficients.append(coeffs)
    # 4. Volumes => minimum and maximum values
    coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
    coefficients.append(coeffs)

    # # 5. MACD => minimum and maximum values
    # coeffs = (np.min(data['MACD'])/margin, np.max(data['MACD'])*margin)
    # coefficients.append(coeffs)

    # # 6. CCI => minimum and maximum values
    # coeffs = (np.min(data['CCI'])/margin, np.max(data['CCI'])*margin)
    # coefficients.append(coeffs)

    # # 7. EMA20 => minimum and maximum values
    # coeffs = (np.min(data['EMA20'])/margin, np.max(data['EMA20'])*margin)
    # coefficients.append(coeffs)

    # # 8. ATR => minimum and maximum values
    # coeffs = (np.min(data['ATR'])/margin, np.max(data['ATR'])*margin)
    # coefficients.append(coeffs)

    return coefficients


def getEpsilonValue(iteration):
    epsilonStart = constants['epsilonGreedy']['start']
    epsilonEnd = constants['epsilonGreedy']['end']
    epsilonDecay = constants['epsilonGreedy']['decay']
    return epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)


def processReward(reward):
    rewardClipping = constants['rewardClipping']
    return np.clip(reward, -rewardClipping, rewardClipping)


def computeLowerBound(cash, numberOfShares, price):
    # Computation of the RL action lower bound
    epsilon = 0.01
    transactionsCost = constants['transactionsCost']

    deltaValues = -cash - numberOfShares * price * \
        (1 + epsilon) * (1 + transactionsCost)

    if deltaValues < 0:
        return deltaValues / (price * (2 * transactionsCost + epsilon * (1 + transactionsCost)))
    return deltaValues / (price * (1 + transactionsCost))


def computeVolatility(data):
    data = data
    return np.std(data) / np.mean(data)


def generateData(data, options):
    volatility = computeVolatility(data)
    runs = []
    options['count'] = options['count'] if options['count'] else len(data)
    options['epochs'] = options['epochs'] if options['epochs'] else 5
    options['min'] = options['min'] if options['min'] else None
    options['max'] = options['max'] if options['max'] else None
    for i in range(0, options['epochs']):
        randomData = [data[-1]]
        for i in range(0, options['count']):
            random = np.random.random()
            changePercent = 2 * volatility * random
            if (changePercent > volatility):
                changePercent -= 2 * volatility
            prevValue = randomData[-1]
            # 50% chance to take second last data point
            if (random < 0.5 and len(randomData) > 1):
                prevValue = randomData[-2]
            changeAmount=prevValue * changePercent
            changeVal=prevValue + changeAmount
            if (options['min'] and changeVal <= options['min']):
                changeVal=options['min']
            if (options['max'] and changeVal > options['max']):
                changeVal=options['max']
            randomData.append(changeVal)
        runs.append(randomData)

    finalRandomData=[]

    for i in range(0, options['count']):
        s=0
        for j in range(0, options['epochs']):
            s += runs[j][i]
        offset=1
        s += data[(offset * i) % len(data)]
        # s += data[((offset + 1) * i) % len(data)]
        # s += data[((offset + 2) * i) % len(data)]
        finalRandomData.append(s / (options['epochs'] + 3))

    return finalRandomData


def generateTrainDataFromDF(dataframe):
    # create new empty dataframe called df
    options={
        'count': 500,
        'epochs': 50,
        'min': None,
        'max': None
    }

    opens = dataframe['Open'].tolist()
    opens = opens + generateData(opens, options)

    closes = dataframe['Close'].tolist()
    closes = closes + generateData(closes, options)

    highs = dataframe['High'].tolist()
    highs = highs + generateData(highs, options)

    lows = dataframe['Low'].tolist()
    lows = lows + generateData(lows, options)

    volumes = dataframe['Volume'].tolist()
    volumes = volumes + generateData(volumes, {
        'count': 500,
        'epochs': 100,
        'min': 0,
        'max': None
    })

    df = pd.DataFrame()
    df['Open'] = opens
    df['Close'] = closes
    df['High'] = highs
    df['Low'] = lows
    df['Volume'] = volumes

    return df
