import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import models
import utils
import os
import time

class PGAN(object):

    def __init__(self, dataloader, args):

        self.dataloader = dataloader
        self.args = args
        self.kimgs = 0
        self.res = 2
        self.batchSize = {2:128, 3:128, 4:128, 5:64, 6:32, 7:16, 8:8}
        self.kticks = {2:100, 3:100, 4:80, 5:60, 6:60, 7:40, 8:20}
        self.device = torch.device("cuda:0")
        
        # Forming output folders
        self.out_imgs = os.path.join(args.outf, 'images')
        self.out_checkpoints = os.path.join(args.outf, 'checkpoints')
        utils.mkdirp(self.out_imgs)
        utils.mkdirp(self.out_checkpoints)
        
        # Defining networks and optimizers
        self.netG = models.Generator()
        self.netD = models.Discriminator()
        
        print('Generator')
        print(self.netG)
        
        print('Discriminator')
        print(self.netD)
        
        # Weight initialization
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion = nn.BCELoss()

        self.netD.cuda()
        self.netG.cuda()
        self.criterion.cuda()

        # Defining optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

        # Other variables
        self.real_label = 1
        self.fake_label = 0
        self.fixed_noise = torch.randn(8, self.args.nz, 1, 1, device=self.device)

    def Train(self, res):

        if res==2:
        	print('Start training')
        	self.start_time = time.time()
        
        start_kimgs = self.kimgs
        if res == self.args.resolution:
            end_kimgs = self.args.kimgs_total
        else:
            end_kimgs = self.kimgs + (self.args.kimgs_training + self.args.kimgs_transition)
        
        epoch = 0
        print('########################')
        print('Training resolution: %d' %res)
    
        while(self.kimgs<end_kimgs):
            epoch += 1
            for i, data in enumerate(self.dataloader, 0):
            
                if self.kimgs>end_kimgs:
                	break
                	
                alpha = (self.kimgs-start_kimgs)/self.args.kimgs_transition
                if alpha>1:
                    alpha = 1
    
                real_data = data[0].to(self.device)
                batch_size = real_data.size(0)
                self.kimgs = self.kimgs + float(batch_size)/1000

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.netD.zero_grad()
                label = torch.full((batch_size,), self.real_label, device=self.device)  
                output = self.netD(real_data, res, alpha)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, self.args.nz, 1, 1, device=self.device)
                fake = self.netG(noise, res, alpha)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach(), res, alpha)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.netD(fake, res, alpha)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if int(self.kimgs/self.kticks[res]) != int((self.kimgs - float(batch_size)/1000)/self.kticks[res]):
                    
                    time_elapsed = utils.disp_time(time.time() - self.start_time)
                    print('%s kimgs: %f Epoch: %d Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (time_elapsed, self.kimgs, epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    
                    scale_factor = int(2 ** (self.args.resolution-res))
                    if scale_factor == 1:
                    	real_data_disp = real_data
                    else:
                    	real_data_disp = F.upsample(real_data, scale_factor=scale_factor, mode='nearest')
                    vutils.save_image(real_data_disp[0:8, :, :, :],
                            '%s/real_samples_res_%d.png' % (self.out_imgs, res), nrow=4,
                            normalize=True)
                    fake = self.netG(self.fixed_noise, res, alpha)
                    if scale_factor == 1:
                    	fake_disp = fake.detach()
                    else:
                    	fake_disp = F.upsample(fake.detach(), scale_factor=scale_factor, mode='nearest') 
                    vutils.save_image(fake_disp,
                            '%s/gen_samples_res_%d_kimgs_%03d.png' % (self.out_imgs, res, self.kimgs), nrow=4,
                            normalize=True)

                    # do checkpointing
                    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.out_checkpoints, epoch))
                    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.out_checkpoints, epoch))
    
    
