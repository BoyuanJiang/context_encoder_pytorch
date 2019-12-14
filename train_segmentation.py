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
from torch.autograd import Variable

from model import _netG
from model import _semantic

from utils import *

# TODO: Could add submodule Pytorch-Encoding with segmentationLoss and other evaluation implementation. Add to robot-tool repo?
# TODO: delete useless code after fixing all the TODOs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/val', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

# free gpu memory
torch.cuda.empty_cache() 

# generator network
netG = _netG(opt)
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
# evaluation mode, no dropout batchnorm stuff. Still freeze it in case.
for param in netG.parameters():
    param.requires_grad = False
netG.eval()

#TODO: initialize weights later
semantic = _semantic(opt)

#TODO: incorporate ground truth labels with dataloader. Lezhou?
transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                    # transforms.CenterCrop(opt.imageSize,
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.ToTensor()])
# dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
dataset = ImageDataset(input_dir='dataset/train/release1_seq1/', transformer=transform)
print(dataset.__getitem__(1))
print(dataset.__len__())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize/2, opt.imageSize/2)

criterionMSE = nn.MSELoss()
optimizer_fc = optim.Adam(semantic.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.cuda:
    print("Using GPU...")
    netG.cuda()
    semantic.cuda()
    input_real, input_cropped = input_real.cuda(),input_cropped.cuda()
    criterionMSE.cuda()
    real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
real_center = Variable(real_center)

for epoch in range(opt.niter):
    running_loss = 0.0
    # dataiter = iter(dataloader)
    for idx, data in enumerate(dataloader, 0):
        # label not included in dataloader yet
        real_cpu, label = data
        if opt.cuda:
            label = label.cuda()
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center_cpu = real_cpu[:, :, opt.imageSize / 4:opt.imageSize / 4 + opt.imageSize / 2,
                          opt.imageSize / 4:opt.imageSize / 4 + opt.imageSize / 2]
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

        input_cropped.data[:, 0,
        opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred,
        opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred] = 2 * 117.0 / 255.0 - 1.0
        input_cropped.data[:, 1,
        opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred,
        opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred] = 2 * 104.0 / 255.0 - 1.0
        input_cropped.data[:, 2,
        opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred,
        opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred] = 2 * 123.0 / 255.0 - 1.0

        # construct a new network
        # TODO: One issue need to be discussed, use real image or cropped image
        representation = netG.getBottleneck(input_cropped)
        print(representation.size())
        output = semantic.forward(representation)

        # using regular MSE loss here
        err = criterionMSE(output, label)

        err.backward()
        optimizer_fc.step()

        running_loss += loss.cpu().item()

print("finished training")

PATH = './fc.pth'
torch.save(semantic.state_dict(), PATH)


# from psnr import psnr
# import numpy as np
#
# t = real_center - fake
# l2 = np.mean(np.square(t))
# l1 = np.mean(np.abs(t))
# real_center = (real_center+1)*127.5
# fake = (fake+1)*127.5
#
# for i in range(opt.batchSize):
#     p = p + psnr(real_center[i].transpose(1,2,0) , fake[i].transpose(1,2,0))
#
# print(l2)
#
# print(l1)
#
# print(p/opt.batchSize)



