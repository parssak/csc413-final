from tqdm import tqdm
from constants import constants
from utils import utils
from utils.data_augmentation import DataAugmentation
from utils.documentor import Documentor
from utils.replay_memory import ReplayMemory
from utils.performance_estimator import PerformanceEstimator
from ml.dqn import DQN
from trading_environment import TradingEnvironment
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np


class TDQN():
    def __init__(self):
        self.device = torch.device(
            'cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')
        self.iterations = 0
        self.observationSpace = constants['observationSpace']
        self.actionSpace = constants['actionSpace']
        self.policyNetwork = DQN(self.observationSpace, self.actionSpace)
        self.targetNetwork = DQN(self.observationSpace, self.actionSpace)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()

        self.optimizer = optim.Adam(self.policyNetwork.parameters(
        ), lr=constants['learningRate'])

        self.replayMemory = ReplayMemory(constants['memory']['capacity'])

    def processState(self, state, coefficients):
        # Normalization of the RL state
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        # 1. Close price => returns => MinMax normalization
        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1]
                   for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0])/(coefficients[0]
                         [1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]

        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [abs(highPrices[i]-lowPrices[i])
                      for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0])/(coefficients[1]
                         [1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]

        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i]-lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i]-lowPrices[i])/deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0])/(coefficients[2]
                         [1] - coefficients[2][0])) for x in closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]

        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0])/(coefficients[3]
                         [1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

        # 5. MACD => MinMax normalization
        # macd = [state[4][i] for i in range(len(state[4]))]
        # if coefficients[4][0] != coefficients[4][1]:
        #     state[4] = [((x - coefficients[4][0])/(coefficients[4]
        #                  [1] - coefficients[4][0])) for x in macd]
        # else:
        #     state[4] = [0 for x in macd]

        # # 6. CCI => MinMax normalization
        # cci = [state[5][i] for i in range(len(state[5]))]
        # if coefficients[5][0] != coefficients[5][1]:
        #     state[5] = [((x - coefficients[5][0])/(coefficients[5]
        #                  [1] - coefficients[5][0])) for x in cci]
        # else:
        #     state[5] = [0 for x in cci]

        # # 7. EMA20 => MinMax normalization
        # ema20 = [state[6][i] for i in range(len(state[6]))]
        # if coefficients[6][0] != coefficients[6][1]:
        #     state[6] = [((x - coefficients[6][0])/(coefficients[6]
        #                  [1] - coefficients[6][0])) for x in ema20]
        # else:
        #     state[6] = [0 for x in ema20]

        # # 8. ATR => MinMax normalization
        # atr = [state[7][i] for i in range(len(state[7]))]
        # if coefficients[7][0] != coefficients[7][1]:
        #     state[7] = [((x - coefficients[7][0])/(coefficients[7]
        #                  [1] - coefficients[7][0])) for x in atr]
        # else:
        #     state[7] = [0 for x in atr]

        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]
        return state

    def training(self, trainingEnv, testData):
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        numEpisodes = constants['numberOfEpisodes']

        # Training performance
        performanceTrain = []
        score = np.zeros((len(trainingEnvList), numEpisodes))

        # Testing performance
        testingEnv = TradingEnvironment(testData)

        # bestPerformanceValue = 0
        # bestWeights = None

        performanceValues = []
        weights = []
        i = 0 

        try:
            for episode in tqdm(range(numEpisodes), disable=not(constants['verbose'])):
                for index, env in enumerate(trainingEnvList):
                    trainingEnvList[index] = self.trainEnvironment(env)

                # Testing performance
                # trainingEnv = self.testing(trainingEnv, trainingEnv)
                # analyzer = PerformanceEstimator(trainingEnv.data)
                # performance = analyzer.computeSharpeRatio()
                # performanceTrain.append(performance)
                # print("performance train: ", performance, analyzer.computeAnnualizedReturn())
                # trainingEnv.reset()

                # * Take the best model
                if (constants['takeBest'] and (numEpisodes-i-1) <= constants['takeBestRange']):
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyzer = PerformanceEstimator(testingEnv.data)
                    performance = analyzer.computeSharpeRatio()
                    performanceValues.append(performance)
                    weights.append(self.policyNetwork.state_dict())
                    testingEnv.reset()

                i += 1
        except KeyboardInterrupt:
            print('Training interrupted prematurely.')

        # Save the best performing model
        if (constants['takeBest'] and len(performanceValues) > 0):
            # get maximum value from performance array
            index = performanceValues.index(max(performanceValues))
            # get the weights of the best performing model
            bestWeights = weights[index]
            self.policyNetwork.load_state_dict(bestWeights)
        elif (constants['takeBest'] and len(performanceValues) == 0):
            print("Performance values not saved...")

        trainingEnvironment = self.testing(trainingEnv, trainingEnv)

        return trainingEnvironment

    def trainEnvironment(self, env):
        coefficients = utils.getNormalizedCoefficients(env)
        env.reset()
        startingPoint = random.randrange(len(env.data.index))
        env.setStartingPoint(startingPoint)
        state = self.processState(env.state, coefficients)
        previousAction = 0
        done = 0
        stepsCounter = 0
        self.hidden_states = self.policyNetwork.init_hidden_states()

        while not done:
            action = self.chooseActionEpsilonGreedy(state, previousAction)
            nextState, reward, done, info = env.step(action)
            reward = utils.processReward(reward)
            nextState = self.processState(nextState, coefficients)
            self.replayMemory.push(state, action, reward, nextState, done)

            # Better exploration trick
            otherAction = int(not bool(action))
            otherReward = utils.processReward(info['Reward'])
            otherNextState = self.processState(
                info['State'], coefficients)
            otherDone = info['Done']
            self.replayMemory.push(
                state, otherAction, otherReward, otherNextState, otherDone)

            # Execute the DQN learning procedure
            stepsCounter += 1
            if stepsCounter == constants['learningUpdatePeriod'] and len(self.replayMemory) >= constants['memory']['batchSize']:
                self.learning()
                stepsCounter = 0

            state = nextState
            previousAction = action

        return env

    def learning(self):
        self.policyNetwork.train()

        # Sample a batch of experiences from the replay memory
        state, action, reward, nextState, done = self.replayMemory.sample()

        # Initialization of Pytorch tensors for the RL experience elements
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(
            reward, dtype=torch.float, device=self.device)
        nextState = torch.tensor(
            nextState, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device)

        # Compute the current Q values returned by the policy network
        result = self.policyNetwork(
            state, self.hidden_states[0], self.hidden_states[1])
        self.hidden_states = result[1]

        currentQValues = result[0].gather(
            1, action.unsqueeze(1)).squeeze(1)

        # Compute the next Q values returned by the target network
        with torch.no_grad():
            result = self.policyNetwork(
                nextState, self.hidden_states[0], self.hidden_states[1])
            nextActions = torch.max(result[0], 1)[1]
            nextQValues = self.targetNetwork(nextState, self.hidden_states[0], self.hidden_states[1])[0].gather(
                1, nextActions.unsqueeze(1)).squeeze(1)
            expectedQValues = reward + \
                constants['gamma'] * nextQValues * (1 - done)

        # Compute the Huber loss
        loss = F.smooth_l1_loss(currentQValues, expectedQValues)
        Documentor.instance().add_loss(loss.item())

        # Computation of the gradients
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            self.policyNetwork.parameters(), constants['gradientClipping'])
        # Perform the Deep Neural Network optimization
        self.optimizer.step()
        # If required, update the target deep neural network (update frequency)
        self.updateTargetNetwork()
        # Set back the Deep Neural Network in evaluation mode
        self.policyNetwork.eval()

    def chooseAction(self, state):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.

        INPUTS: - state: RL state returned by the environment.

        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            tensorState = torch.tensor(
                state, dtype=torch.float, device=self.device).unsqueeze(0)
            result = self.policyNetwork(
                tensorState, self.hidden_states[0], self.hidden_states[1])
            self.hidden_states = result[1]
            QValues = result[0].squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues

    def chooseActionEpsilonGreedy(self, state, previousAction):
        if(random.random() > utils.getEpsilonValue(self.iterations)):
            if(random.random() > constants['epsilonGreedy']['alpha']):
                action, Q, QValues = self.chooseAction(state)
            else:
                action = previousAction
                Q = 0
                QValues = [0, 0]
        else:
            action = random.randrange(constants['actionSpace'])
            Q = 0
            QValues = [0, 0]

        self.iterations += 1

        return action

    def updateTargetNetwork(self):
        if(self.iterations % constants['targetNetworkUpdate'] == 0):
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    def testing(self, trainingEnv, testingEnv, storeResults=False):
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(
            testingEnv, constants['filterOrder'])
        trainingEnv = dataAugmentation.lowPassFilter(
            trainingEnv, constants['filterOrder'])

        # Initialization of some RL variables
        coefficients = utils.getNormalizedCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        q_values = []
        done = 0
        self.hidden_states = self.policyNetwork.init_hidden_states()

        # Interact with the environment until the episode termination
        while not done:
            action, _, QValues = self.chooseAction(state)

            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)

            # Update the new state
            state = self.processState(nextState, coefficients)

            # Storing of the Q values
            q_values.append([float(QValues[0]), float(QValues[1])])

        if storeResults:
            d = Documentor.instance()
            d.write('qValues', q_values)

        return testingEnv
