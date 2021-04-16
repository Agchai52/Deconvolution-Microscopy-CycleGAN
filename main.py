"""
Main file to call for train or test:
1. load all options for train or test
2. select gpu or cpu to train or test
"""
from __future__ import print_function  # help to use print() in python 2.x
import argparse
import os
import torch.backends.cudnn as cudnn

import train
import test
from network import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='DeblurMicrocope', help='DeblurMicrocope')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of training epochs')
parser.add_argument('--epoch_start', dest='epoch_start', type=int, default=0, help='# of start epoch')
parser.add_argument('--epoch_decay', dest='epoch_decay', type=int, default=0, help='# of epoch to linearly decay lr to 0')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--load_size', dest='load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=64, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=5.0, help='weight on L1 term in objective')
parser.add_argument('--dark_channel_lambda', dest='dark_channel_lambda', type=float, default=100, help='weight on Dark Channel loss in objective')
parser.add_argument('--H', dest='H', default=256, type=int, help='Test size H')
parser.add_argument('--W', dest='W', default=256, type=int, help='Test size W')
parser.add_argument('--gpu', '-g', default=1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--seed', type=int, default=123, help='random seed to use, Default=123')
parser.add_argument('--save_intermediate', dest='save_intermediate', type=bool, default=True, help='Save validation image and metrics each epoch, Default=False')

args = parser.parse_args()
print(args)

cudnn.benchmark = True

torch.manual_seed(args.seed)

if args.phase == 'train':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs("checkpoint/netG")
        os.makedirs("checkpoint/netD_B")
        os.makedirs("checkpoint/netD_S")

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    train.train(args)

elif args.phase == 'test':
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    test.test(args)

elif args.phase == 'test_real':
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    test.test_real(args)
else:
    raise Exception("Phase should be 'train" or 'test' or 'test_real')

