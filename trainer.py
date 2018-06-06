import torch
import torch.nn as nn
import torch.optim as optim
import models
import utils
import os

class PGAN(object):

    def __init__(self, dataloader, args):

        self.dataloader = dataloader
        self.args = args
        self.kimgs = 0
        self.res = 2
        self.batchSize = {2:128, 3:128, 4:64, 5:64, 6:32, 7:16, 8:16}
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

    def Train(self, res):

        start_kimgs = self.kimgs
        if res == self.args.resolution:
            end_kimgs = self.kimgs_total
        else:
            end_kimgs = self.kimgs + (self.args.kimgs_training + self.args.kimgs_transition)
        
        epoch = 0
        print('########################')
        print('Training resolution: %d' %res)
    
        while(self.kimgs<end_kimgs):
            epoch += 1
            for i, data in enumerate(self.dataloader, 0):
            
                alpha = (self.kimgs-start_kimgs)/self.args.kimgs_transition
                if alpha>1:
                    alpha = 1
    
                real_data = data[0].to(self.device)
                batch_size = real_data.size(0)
                self.kimgs = self.kimgs + batch_size/1000

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.netD.zero_grad()
                label = torch.full((batch_size,), self.real_label, device=self.device)  
                output = self.netD(real_data, res, alpha)
                print(output.size())
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, self.args.nz, 1, 1, device=device)
                fake = self.netG(noise, res, alpha)
                label.fill_(fake_label)
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
                label.fill_(real_label)  # fake labels are real for generator cost
                output = self.netD(fake, res, alpha)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if self.kimgs%self.kticks[res] == 0:
                    
                    print('kimgs: %d Epoch: %d Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (self.kimgs, epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    
                    vutils.save_image(real_data,
                            '%s/real_samples.png' % self.out_imgs,
                            normalize=True)
                    fake = netG(self.fixed_noise, res, alpha)
                    vutils.save_image(fake.detach(),
                            '%s/gen_samples_kimgs_%03d.png' % (self.out_imgs, self.kimgs),
                            normalize=True)

                    # do checkpointing
                    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.out_checkpoints, epoch))
                    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.out_checkpoints, epoch))
    
    
