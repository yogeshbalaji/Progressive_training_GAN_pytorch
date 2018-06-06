from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

### Default parameters 
nc = 3
nz = 100
fmap_base = 8192
fmap_decay = 1.0
fmap_max = 512

def nf(stage):
    return min(int(fmap_base/ (2.0 ** (stage*fmap_decay))), fmap_max)

def GBlock(res):
    if res == 2:
        block = nn.Sequential(
            nn.ConvTranspose2d(nz, nf(res-1), 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf(res-1), nf(res-1), 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True)
        )
    else:
        block = nn.Sequential(
            nn.ConvTranspose2d(nf(res-2), nf(res-1), 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf(res-1), nf(res-1), 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True)
        )
    return block

def GtoRGB(res):
    block = nn.Sequential(
        nn.Conv2d(nf(res-1), nc, 1, bias=False)
    )
    return block
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.upscale = nn.Upsample(scale_factor=2)
        
        self.block = []
        self.toRGB = []
        
        for res in range(2, 9):
            self.block.append(GBlock(res))
            self.toRGB.append(GtoRGB(res))
        
        self.block = nn.Sequential(*self.block)
        self.toRGB = nn.Sequential(*self.toRGB)
    def forward(self, noise, res, alpha):
        
        noise = noise.view(-1, nz, 1, 1)
        
        for i in range(2, res):
            if i==2:
                x = self.block[i-2](noise)
            else:
                x = self.upscale(x)
                x = self.block[i-2](x)
        
        if res == 2:
            x = self.block[res-2](noise)
            x = self.toRGB[res-2]
        else:
            y = self.toRGB[res-3](x)
            y = self.upscale(y)
            
            z = self.upscale(x)
            z = self.block[res-2](z)
            z = self.toRGB[res-2]
            
            if alpha == 1:      # No fading
                x = z
            else:               # Fading
                x = (1-alpha)*y + alpha*z
        
        return x
        
##########################################################

def DBlock(res):
    if res == 2:
        block = nn.Sequential(
            nn.Conv2d(nf(res-1), nf(res-1), 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True),
            nn.Conv2d(nf(res-1), nf(res-2), 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True),
            nn.Conv2d(nf(res-1), 1, 1, bias=False)
        )
    else:
        block = nn.Sequential(
            nn.ConvTranspose2d(nf(res-1), nf(res-1), 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf(res-1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf(res-1), nf(res-2), 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf(res-2)),
            nn.ReLU(True)
        )
    return block

def DfromRGB(res):
    block = nn.Sequential(
        nn.Conv2d(nc, nf(res-1), 1, bias=False)
    )
    return block
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.downscale = nn.AvgPool2d(2, 2)
        
        self.block = []
        self.fromRGB = []
        
        for res in range(2, 9):
            self.block.append(DBlock(res))
            self.fromRGB.append(DfromRGB(res))
        
        self.block = nn.Sequential(*self.block)
        self.fromRGB = nn.Sequential(*self.fromRGB)
        
    def forward(self, imgs, res, alpha):
        
        if res == 2:
            x = self.fromRGB[res-2](imgs)
            x = self.block[res-2](x)
        else:
            y = self.downscale(imgs)
            y = self.fromRGB[res-3](y)
            
            z = self.fromRGB[res-2](imgs)
            z = self.block[res-2](z)
            z = self.downscale(z)
            
            if alpha == 1:
                x = z
            else:
                x = (1-alpha)*y + alpha*z
            
        for i in range(res-1, 1, -1):
            if i==2:
                x = self.block[i-2](x)
            else:
                x = self.block[i-2](x)
                x = self.downscale(x)
        
        return x


##########################################################
        

