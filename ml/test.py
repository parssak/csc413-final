from ml.sagan import Generator, Discriminator
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from utils.data_handler import DataHandler
# import np
import numpy as np

class DataPasser():
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size
        self.pointer = 0

    def next_batch(self):
        if self.pointer + self.batch_size > len(self.data):
            self.pointer = 0
        batch = self.data[self.pointer:self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return batch


class Env():
    def __init__(self):
        data_handler = DataHandler()
        data = data_handler.get_data()
        data = np.array([data['Open'], data['High'], data['Low'], data['Close']])
        data = data.transpose()
        self.batch_size = 64
        print(data.shape)
        self.data_passer = DataPasser(data, self.batch_size)
        self.input_size = 64
        self.z_dim = 100
        self.g_conv_dim = 64
        self.d_lr = 2e-4
        self.g_lr = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

    def build_model(self):
        self.D = Discriminator()
        self.G = Generator(self.batch_size)
        self.d_optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])

        self.c_loss = nn.CrossEntropyLoss()
        print('Model built')
        # print(self.D)
        # print(self.G)

    def train(self):
        numEpisodes = 50

        for episode in tqdm(range(numEpisodes)):
            self.D.train()
            self.G.train()

            print('?')
            for d_epoch in range(5):
                print('!')
                self.D.zero_grad()

                # 1. Train D on real+fake
                real_input = torch.from_numpy(self.data_passer.next_batch()).float()
                d_real = self.D(real_input)
                d_real_loss = -torch.mean(d_real)

                z = torch.randn(self.batch_size, self.z_dim)
                d_fake = self.D(self.G(z))
                d_fake_loss = torch.mean(d_fake)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                
                self.d_optimizer.step()

                # 2. Train G
                self.G.zero_grad()

                z = torch.randn(self.batch_size, self.z_dim)
                d_fake = self.D(self.G(z))
                g_loss = -torch.mean(d_fake)
                g_loss.backward()

                self.g_optimizer.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                episode, numEpisodes, d_epoch, 5, d_loss.data[0], g_loss.data[0]))

def test():
    env = Env()
    env.build_model()
    env.train()
