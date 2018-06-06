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
import utils
import trainer

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--resolution', type=int, default=8, help='Resolution of images to train is 2^resolution')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--kimgs_total', type=int, default=15000, help='number of kimgs to train for')
    parser.add_argument('--kimgs_training', type=int, default=600, help='number of kimgs to train after fading')
    parser.add_argument('--kimgs_transition', type=int, default=600, help='number of kimgs to fade')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    
    # Setting random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    cudnn.benchmark = True

    # Creating trainer class
    PGAN = trainer.PGAN('', args)
    
    for res in range(2, args.resolution+1):
        img_size = int(2 ** res)
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                     transforms.Resize(img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=PGAN.batchSize[res],
                                         shuffle=True, num_workers=int(args.workers))
        PGAN.dataloader = dataloader
        PGAN.Train(res)
        

if __name__ == '__main__':
    main()

