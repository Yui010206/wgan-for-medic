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
from torchvision.utils import save_image
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

netG = Generator(DIM)
netD = Discriminator(DIM)
netG = torch.load('G.pt')

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

for i in range(10):
    noise = torch.randn(1, 128)
    noise = noise.cuda()
    fake = netG(noise).data
    save_image(fake, 'output/%04d.png' % (i+1))






