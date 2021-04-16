import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class DeblurDataset(Dataset):
    def __init__(self, img_path, args, is_train=True):
        self.img_path = img_path
        self.args = args
        self.is_train = is_train
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])

    def __getitem__(self, index):
        '''
        Transforms:
        1. Resize image to (fine_size, fine_size)  for  CycleGAN fine_size = 64
        2. random flip image from left to right
        3. PIL.Image(H,W,C) to Tensor(C,H,W)
        4. normalize from [0.0, 1.0] to [-1.0, 1.0]
        '''

        img_A = Image.open(self.img_path[index] + '_blur.png').convert('L')
        img_B = Image.open(self.img_path[index] + '_sharp.png').convert('L')
        img_name = self.img_path[index][-4:]
        img_name.rstrip()

        # w = int(img_A.size[0])
        # h = int(img_A.size[1])

        img_A = transforms.Resize((self.args.fine_size, self.args.fine_size))(img_A)

        if self.is_train:  # Only resize sharp and flip sharp and blurry for training
            img_B = transforms.Resize((self.args.fine_size, self.args.fine_size))(img_B)
            if np.random.random() < 0.5:
                img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
                img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() < 0.5:
                img_A = img_A.transpose(Image.FLIP_TOP_BOTTOM)
                img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A, img_B, img_name

    def __len__(self):
        return len(self.img_path)

