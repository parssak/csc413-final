def computeMACD(dataframe):
    exp1 = dataframe['Close'].ewm(span=12, adjust=False).mean()
    exp2 = dataframe['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd


def computeCCI(dataframe):
    tp = (dataframe['High'] + dataframe['Low'] + dataframe['Close']) / 3
    sma = tp.rolling(window=14).mean()
    mad = tp.rolling(window=14).apply(lambda x: x.mad())
    print('##')
    print(mad.max(), mad.min())
    cci = (tp - sma) / (0.015 * mad)
    cci[cci == float('inf')] = 0  # replace all inf with 0
    return cci


def computeSMI(dataframe):
    # compute the stochastic momentum index
    smi = (dataframe['Close'] - dataframe['Low'].rolling(window=14).min()) / \
        (dataframe['High'].rolling(window=14).max() -
         dataframe['Low'].rolling(window=14).min())
    return smi


def computeEMA20(dataframe):
    ema20 = dataframe['Close'].ewm(span=20, adjust=False).mean()
    return ema20


def computeATR(dataframe):
    # compute the average true range
    atr = dataframe['High'].rolling(window=14).max() - \
        dataframe['Low'].rolling(window=14).min()
    atr = atr.rolling(window=14).mean()
    return atr


def applyIndicators(dataframe):
    dataframe['MACD'] = computeMACD(dataframe)
    dataframe['CCI'] = computeCCI(dataframe)
    dataframe['SMI'] = computeSMI(dataframe)
    dataframe['EMA20'] = computeEMA20(dataframe)
    dataframe['ATR'] = computeATR(dataframe)
    return dataframe
