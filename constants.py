def getObservationSpace(stateLength, numInputTypes):
    return 1 + (stateLength-1)*numInputTypes


stateLength = 30

constants = {
    "symbol": "apple",
    "dates": {
        "start": "2012-01-03",
        "end": "2019-12-31",
        "splitPct": 0.7
    },
    # "symbol": "btc",
    # "dates": {
    #     "start": "2021-01-07 13:00",
    #     "end": "2021-03-15 20:00",
    #     "splitPct": 0.3
    # },
    "takeBest": True,
    "takeBestRange": 4,
    "numberOfEpisodes": 50,
    "money": 1000000,
    "transactionsCost": 0.001,
    "stateLength": stateLength,
    "observationSpace": 117,  # add 30 when adding another indicator
    "actionSpace": 2,
    "bounds": [1, 30],
    "step": 1,
    "gamma": 0.4,
    "learningRate": 0.0001,
    "targetNetworkUpdate": 1000,
    "learningUpdatePeriod": 1,
    "numberOfNeurons": 512,
    "dropout": 0.2,
    "memory": {
        "capacity": 100000,
        "batchSize": 32,
        "experiencesRequired": 1000,
    },
    "epsilonGreedy": {
        "start": 1.0,
        "end": 0.01,
        "decay": 10000,
        "alpha": 0.2,
    },
    "rewardClipping": 1,
    "gradientClipping": 1,
    "dataAugmentation": {
        "shiftRanges": [0],
        "stretchRanges": [1],
        "filterRanges": [5],
        "noiseRanges": [0]
    },
    "filterOrder": 5,
    "verbose": True,
    # Custom parameters
    "maxValidPassiveRange": 336,
    "frequencyObservationWindow": 50,
    "frequencyActionCutoff": 0.7,
}
