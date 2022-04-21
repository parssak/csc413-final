from constants import constants
from ml.il import Discriminator
from utils import utils
from matplotlib import pyplot as plt
import os
import gym
import math
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

from utils.data_handler import DataHandler

pd.options.mode.chained_assignment = None


class TradingEnvironment(gym.Env):
    def __init__(
        self,
        data,
        money=constants['money'],
        stateLength=constants['stateLength'],
        transactionCosts=constants['transactionsCost'],
        startingPoint=0
    ):
        self.data = data
        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5,
                              limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.stateLength = stateLength
        self.state = self._getStateArray(0, self.stateLength, 0)
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0
        self.transactionCosts = constants['transactionsCost']

        self.discriminator = Discriminator(150, 128, 2)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.001)

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)

    def reset(self):
        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = self._getStateArray(0, self.stateLength, 0)
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    def step(self, action):
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False

        # CASE 1: LONG POSITION
        if(action == 1):
            self.data['Position'][t] = 1
            # Case a: Long -> Long
            if(self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * \
                    self.data['Close'][t]
            # Case b: No position -> Long
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(
                    self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * \
                    self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * \
                    self.data['Close'][t]
                self.data['Action'][t] = 1
            # Case c: Short -> Long
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * \
                    self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(
                    self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * \
                    self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * \
                    self.data['Close'][t]
                self.data['Action'][t] = 1

        # CASE 2: SHORT POSITION
        elif(action == 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if(self.data['Position'][t - 1] == -1):
                lowerBound = utils.computeLowerBound(
                    self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] = - \
                        self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(
                        lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * \
                        self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] = - \
                        self.numberOfShares * self.data['Close'][t]
                    customReward = True
            # Case b: No position -> Short
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(
                    self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * \
                    self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - \
                    self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
            # Case c: Long -> Short
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * \
                    self.data['Close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(
                    self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * \
                    self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - \
                    self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit(
                "Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        returns = (self.data['Money'][t] -
                   self.data['Money'][t-1])/self.data['Money'][t-1]

        self.data['Returns'][t] = returns

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['Close'][t-1] -
                           self.data['Close'][t])/self.data['Close'][t-1]

        self.reward = self.reward * \
            self._computeHoldingRewardMultiplier(t, self.reward)
        self.reward = self.reward * \
            self._computeTradeFrequencyMultiplier(t, self.reward)

        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = self._getStateArray(
            self.t - self.stateLength, self.t, self.data['Position'][self.t - 1])

        d_reward = self.stepDiscriminator()

        fixed_d_reward = d_reward.item() - 0.25

        # add IL reward
        self.reward = self.reward + fixed_d_reward

        if(self.t == self.data.shape[0]):
            self.done = 1

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * self.data['Close'][t]
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(
                    self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * \
                    self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * \
                    self.data['Close'][t] * (1 + self.transactionCosts)
                numberOfShares = math.floor(
                    otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * \
                    self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
        else:
            otherPosition = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = utils.computeLowerBound(
                    self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings = - numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(
                        math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * \
                        self.data['Close'][t] * (1 + self.transactionCosts)
                    otherHoldings = - numberOfShares * self.data['Close'][t]
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(
                    self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * \
                    self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * \
                    self.data['Close'][t] * (1 - self.transactionCosts)
                numberOfShares = math.floor(
                    otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * \
                    self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'][t]
        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (
                otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] -
                           self.data['Close'][t])/self.data['Close'][t-1]

        otherReward = otherReward * \
            self._computeHoldingRewardMultiplier(t, otherReward)
        otherReward = otherReward * \
            self._computeTradeFrequencyMultiplier(t, otherReward)

        otherState = self._getStateArray(
            self.t - self.stateLength, self.t, otherPosition)
        self.info = {'State': otherState,
                     'Reward': otherReward, 'Done': self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info

    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.

        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments
        self.state = self._getStateArray(
            self.t - self.stateLength, self.t, self.data['Position'][self.t - 1])
        if(self.t == self.data.shape[0]):
            self.done = 1

    def stepDiscriminator(self):
        action = self.state[len(self.state) - 1][0]
        state_copy = self.state.copy()
        state_copy[len(state_copy) - 1] = np.zeros(30)
        state_copy[len(state_copy) - 1][29] = action
                
        tensor_state = torch.tensor(state_copy, dtype=torch.float32)
        flat_state = tensor_state.view(1, -1)

        self.discriminator.train()

        with torch.no_grad():
            self.discriminator.zero_grad()
            output = self.discriminator(flat_state)

        expected = torch.tensor([[0.0, 1.0]], dtype=torch.float32, requires_grad=True)
        da_loss = F.smooth_l1_loss(output, expected)
        da_loss.backward()

        self.discriminator_optimizer.step()

        # now fetch from expert data

        expert = self._getExpertStateArray(self.t - self.stateLength, self.t)
        tensor_state = torch.tensor(expert, dtype=torch.float32)

        with torch.no_grad():
            self.discriminator.zero_grad()
            output = self.discriminator(flat_state)

        expected = torch.tensor([[1.0, 0.0]], dtype=torch.float32, requires_grad=True)

        de_loss = F.smooth_l1_loss(output, expected)
        de_loss.backward()

        self.discriminator_optimizer.step()

        # put back in eval mode

        self.discriminator.eval()

        return da_loss

    def _getStateArray(self, start, end, position):
        return [self.data['Close'][start: end].tolist(),
                self.data['Low'][start: end].tolist(),
                self.data['High'][start: end].tolist(),
                self.data['Volume'][start: end].tolist(),
                # self.data['MACD'][start: end].tolist(),
                # self.data['CCI'][start: end].tolist(),
                # self.data['EMA20'][start: end].tolist(),
                # self.data['ATR'][start: end].tolist(),
                [position]]

    def _getExpertStateArray(self, start, end):
        data = DataHandler().get_expert_data()
        return [data['Close'][start: end].tolist(),
                data['Low'][start: end].tolist(),
                data['High'][start: end].tolist(),
                data['Volume'][start: end].tolist(),
                # data['MACD'][start: end].tolist(),
                # data['CCI'][start: end].tolist(),
                # data['EMA20'][start: end].tolist(),
                # data['ATR'][start: end].tolist(),
                data['Position'][start: end].tolist()
                ]

    def _computeHoldingRewardMultiplier(self, t, reward):
        maxValidPassiveRange = constants['maxValidPassiveRange']
        holdingRewardMultiplier = 1
        if self.data['Action'][t] == 0:
            prevAction = self.data['Action'][t]
            prev = -1
            # find previous Action that is not 0
            for i in range(t-1, 0, -1):
                if self.data['Action'][i] != 0:
                    prevAction = self.data['Action'][i]
                    prev = i
                    break

            distance = t - prev
            if distance > maxValidPassiveRange:
                holdingRewardMultiplier = maxValidPassiveRange / \
                    distance if reward > 0 else distance/maxValidPassiveRange

        return holdingRewardMultiplier

    def _computeTradeFrequencyMultiplier(self, t, reward):
        tradeFrequencyMultiplier = 1
        if self.data['Action'][t] != 0:
            frequencyObservationWindow = constants['frequencyObservationWindow']
            frequencyActionCutoff = constants['frequencyActionCutoff']
            numActionsInWindow = 0
            for i in range(t-1, max(t-frequencyObservationWindow-1, 0), -1):
                if self.data['Action'][i] != 0:
                    numActionsInWindow += 1

            tradingFrequency = numActionsInWindow / frequencyObservationWindow
            if tradingFrequency >= frequencyActionCutoff:
                tradeFrequencyMultiplier = 1 - tradingFrequency

        return tradeFrequencyMultiplier
