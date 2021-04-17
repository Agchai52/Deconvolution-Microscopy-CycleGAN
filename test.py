from __future__ import print_function
import os
import time
import torch
import torchvision.transforms as transforms
from Dataset import DeblurDataset
from torch.utils.data import DataLoader

from utils import *
from network import *
from Dataset import DeblurDataset, RealImage


def test(args):
    print("====> Loading model")
    device = torch.device("cuda")
    model_G = Generator(args, device)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_G = nn.DataParallel(model_G)

    net_g_path = "checkpoint/netG"
    netG = model_G.to(device)
    if not find_latest_model(net_g_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpoint = torch.load(model_path_G)
        netG.load_state_dict(checkpoint['model_state_dict'])
        netG.eval()

    print("====> Loading data")
    ############################
    # For DeblurMicroscope dataset
    ###########################
    f_test = open("./dataset/test_instance_names.txt", "r")
    test_data = f_test.readlines()
    test_data = [line.rstrip() for line in test_data]
    f_test.close()
    test_data_loader = DataLoader(DeblurDataset(test_data, args, False), batch_size=1, shuffle=False)

    ############################
    # For Other datasets
    ###########################
    # image_dir = "dataset/{}/test/a/".format(args.dataset_name)
    # image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
    # transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # transform = transforms.Compose(transform_list)
    # for image_name in image_filenames:

    all_psnr = []
    all_ssim = []
    start_time = time.time()
    with torch.no_grad():
        for batch in test_data_loader:
            real_B, real_S, img_name = batch[0], batch[1], batch[2]
            real_B, real_S = real_B.to(device), real_S.to(device)  # B = (B, 1, 64, 64), S = (B, 1, 256, 256)
            pred_S = netG(real_B)
            # pred_S = F.interpolate(pred_S, (args.load_size, args.load_size), mode='bilinear')  # 64 -> 256
            cur_psnr, cur_ssim = compute_metrics(real_S, pred_S)
            all_psnr.append(cur_psnr)
            all_ssim.append(cur_ssim)
            if img_name[0][-2:] == '01':
                img_S = pred_S.detach().squeeze(0).cpu()
                save_img(img_S, '{}/test_'.format(args.test_dir) + img_name[0])
                print('test_{}: PSNR = {} dB, SSIM = {}'.format(img_name[0], cur_psnr, cur_ssim))

    total_time = time.time() - start_time
    ave_psnr = sum(all_psnr) / len(test_data_loader)
    ave_ssim = sum(all_ssim) / len(test_data_loader)
    ave_time = total_time / len(test_data_loader)
    print("Average PSNR = {}, SSIM = {}, Processing time = {}".format(ave_psnr, ave_ssim, ave_time))

def test_real(args):
    print("====> Loading model")
    device = torch.device("cuda")
    model_G = Generator(args, device)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_G = nn.DataParallel(model_G)

    net_g_path = "checkpoint/netG"
    netG = model_G.to(device)
    if not find_latest_model(net_g_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpoint = torch.load(model_path_G)
        netG.load_state_dict(checkpoint['model_state_dict'])
        netG.eval()

    print("====> Loading data")
    ############################
    # For Real Images
    ###########################
    image_dir = "dataset/{}/".format("real_images")
    image_filenames = [image_dir + x[0:-4] for x in os.listdir(image_dir) if x[-4:] in set([".png", ".jpg"])]
    test_data_loader = DataLoader(RealImage(image_filenames, args, False), batch_size=1, shuffle=False)

    start_time = time.time()
    with torch.no_grad():
        for batch in test_data_loader:
            real_B, img_name = batch[0], batch[1]
            real_B = real_B.to(device)
            pred_S = netG(real_B)
            # pred_S = F.interpolate(pred_S, (real_B.shape[2] * 4, real_B.shape[3] * 4), mode='bilinear')  # (h, w) x 4
            img_S = pred_S.detach().squeeze(0).cpu()
            save_img(img_S, '{}/test_'.format(args.test_dir) + img_name[0])

    total_time = time.time() - start_time
    ave_time = total_time / len(test_data_loader)
    print("Processing time = {}".format(ave_time))
