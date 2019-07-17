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
from utils import *
from models import *

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
DIM = 32 # Model dimensionality
BATCH_SIZE = 1 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 250000 

train_loader = torch.utils.data.DataLoader(dataset=Dataset('train'), batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=Dataset('test'), batch_size=1, shuffle=True)

netG = Generator(DIM)
netD = Discriminator(DIM)
# print(netG)
# print(netD)
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
netG = torch.load("G.pt")
netD = torch.load("D.pt")
optimizerD = optim.Adam(netD.parameters(), lr=5e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=5e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)
    
def inf_train_gen():
    while True:
        for images in train_loader:
            yield images
data = inf_train_gen()

for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():
        p.requires_grad = True
    for q in netG.parameters():
        q.requires_grad = False

    for iter_d in range(CRITIC_ITERS):
        _data = data.__next__()
        real_data = _data
        real_data = real_data.cuda()
        real_data_v = real_data

        netD.zero_grad()

        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(one)

        noise = torch.randn(BATCH_SIZE, 128)
        noise = noise.cuda()
        noisev = noise
        fake = netG(noisev).data

        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(mone)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, BATCH_SIZE)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        if iteration%100 ==0 and iter_d==4:
            print("Processed %d iterations, real loss is %5f, fake loss is %5f, Wasserstein loss is %5f" % (iteration, D_real, D_fake,Wasserstein_D))
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False
    for q in netG.parameters():
        q.requires_grad = True

    netG.zero_grad()
    noise = torch.randn(BATCH_SIZE, 128)
    noise = noise.cuda()
    noisev = noise
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(one)
    G_cost = -G
    optimizerG.step()

    # Calculate dev loss and generate samples every 10000 iters
    if iteration % 10000 == 9999:
        PATH = './'
        torch.save(netG, PATH+'G.pt')
        torch.save(netD, PATH+'D.pt')

    # Write logs every 100 iters
