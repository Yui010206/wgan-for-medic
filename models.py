import os, sys
sys.path.append(os.getcwd())

from PIL import Image
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

class Generator(nn.Module):
    def __init__(self, DIM):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 3*4*4*8*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(8*DIM*3, 8*DIM*3, 7),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(8*DIM*3, 8*DIM*3, 5),
            nn.ReLU(True),
        )
        block3 = nn.Sequential(
            nn.ConvTranspose2d(8*DIM*3, 8*DIM*3, 3),
            nn.ReLU(True),
        )
        ReduceDim = nn.Sequential(
            nn.ConvTranspose2d(8*DIM*3, 4*DIM*3, 3, stride=2),
            nn.ConvTranspose2d(4*DIM*3, 2*DIM*3, 3, stride=2),
            nn.ConvTranspose2d(2*DIM*3, DIM*3, 5, stride=2),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM*3, 3, 1)
        self.DIM = DIM
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.ReduceDim = ReduceDim
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        DIM = self.DIM
        output = self.preprocess(input)
        output = output.view(-1,8*DIM*3, 4, 4)
        output = self.block1(output)
        output = self.block1(output)
        output = self.block1(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block2(output)
        output = self.block2(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block3(output)
        output = self.block3(output)
        output = self.block3(output)
        output = self.ReduceDim(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output
    
    
class Discriminator(nn.Module):
    def __init__(self,DIM):
        super(Discriminator, self).__init__()
        downsample1 = nn.Sequential(
            nn.Conv2d(3, 4*DIM, 7, stride = 2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(4*DIM, 4*DIM, 7, stride = 2),
            nn.LeakyReLU(0.01)
        )
        downsample2 = nn.Sequential(
            nn.Conv2d(4*DIM, 4*DIM, 3, stride = 2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(4*DIM, 4*DIM, 3, stride = 2),
            nn.LeakyReLU(0.01)
        )
        downsample3 = nn.Sequential(
            nn.Conv2d(4*DIM, 4*DIM, 3, stride = 2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(4*DIM, 4*DIM, 3, stride = 2),
            nn.LeakyReLU(0.01)
        )

        self.downsample1 = downsample1
        self.downsample2 = downsample2
        self.downsample3 = downsample3
        self.output = nn.Linear(3200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.downsample1(input)
        out = self.downsample2(out)
        out = self.downsample3(out)
        out = out.view(-1)
        out = self.output(out)
        return out
    
def calc_gradient_penalty(netD, real_data, fake_data, batch_size, use_cuda=True, LAMBDA = 10):
    #print real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty