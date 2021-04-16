import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

def weights_init(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)


class BlurModel(nn.Module):
    def __init__(self, args, device='cpu'):
        super(BlurModel, self).__init__()

        def kernel_fit(loc):
            """
            Estimated psf of laser
            :param loc: (x, y)
            :return: z
            """
            x, y = loc
            x, y = x * 50, y * 50
            z = np.exp(-np.log(2) * (x * x + y * y) / (160.5586 * 160.5586)) * 255
            return z

        def get_kernel():
            """
            Compute cropped blur kernel
            :param is_plot: bool
            :return: blur kernel
            """
            M = 61
            X, Y = np.meshgrid(np.linspace(-30, 31, M), np.linspace(-30, 31, M))
            d = np.dstack([X, Y])
            Z = np.zeros((M, M))
            for i in range(len(d)):
                for j in range(len(d[0])):
                    x, y = d[i][j]
                    Z[i][j] = kernel_fit((x, y))

            Z = Z.reshape(M, M)
            img_Z = np.uint8(np.asarray(Z))
            crop_size = 22
            crop_Z = img_Z[crop_size:M - crop_size, crop_size:M - crop_size]
            kernel = crop_Z / np.float(np.sum(crop_Z))
            return kernel

        self.batch_size = args.batch_size
        self.device = device
        self.kernel = torch.FloatTensor(get_kernel())  # (17, 17)
        self.loss = nn.MSELoss()
        self.kernel_size = self.kernel.shape[0]
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 1, H, W)
        # Padding
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)

        # weight :
        kernel = self.kernel.expand(1, 1, self.kernel_size, self.kernel_size)  # (1, 1, 17, 17)
        kernel = kernel.flip(-1).flip(-2).to(self.device)

        # Convolution
        blur_img = F.conv2d(x, kernel)
        return blur_img


class Generator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Generator, self).__init__()
        self.input_nc = args.input_nc
        self.ngf = args.ngf
        self.device = device
        # Encoder
        self.e1 = nn.Sequential(ConvBlock(self.input_nc, self.ngf * 1),
                                ConvBlock(self.ngf * 1, self.ngf * 1))  # (B, 64, H, W)
        self.e2 = nn.Sequential(nn.MaxPool2d(2),
                                ConvBlock(self.ngf * 1, self.ngf * 2),
                                ConvBlock(self.ngf * 2, self.ngf * 2))  # (B, 128, H/2, W/2)
        self.e3 = nn.Sequential(nn.MaxPool2d(2),
                                ConvBlock(self.ngf * 2, self.ngf * 4),
                                ConvBlock(self.ngf * 4, self.ngf * 4),
                                ConvBlock(self.ngf * 4, self.ngf * 4),  # (B, 256, H/4, W/4)
                                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1))   # (B, 128, H/2, W/2)

        # Decoder
        self.d1 = nn.Sequential(ConvBlock(self.ngf * 4, self.ngf * 2),
                                ConvBlock(self.ngf * 2, self.ngf * 2),  # (B, 128, H/2, W/2)
                                nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1))   # (B, 64, H, W)
        self.d2 = nn.Sequential(ConvBlock(self.ngf * 2, self.ngf * 1),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(self.ngf * 1, self.input_nc, kernel_size=3, stride=1, padding=0,
                                          padding_mode='circular'),     # (B, 1, H, W)
                                nn.Tanh())

    def forward(self, img):
        """
        Layer Sizes after Conv2d for input img size (1, 256, 256) self.ngf = 64
        """
        # Encoder
        e_layer1 = self.e1(img)
        e_layer2 = self.e2(e_layer1)
        e_layer3 = self.e3(e_layer2)

        # Decoder
        e_layer3 = torch.cat([e_layer2, e_layer3], 1)
        d_layer1 = self.d1(e_layer3)

        d_layer1 = torch.cat([e_layer1, d_layer1], 1)
        d_layer2 = self.d2(d_layer1)

        # print(img.shape)
        # print(e_layer1.shape)
        # print(e_layer2.shape)
        # print(e_layer3.shape)
        # print("Decoder")
        # print(d_layer1.shape)
        # print(d_layer2.shape)

        return d_layer2


class Discriminator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Discriminator, self).__init__()
        self.input_nc = args.input_nc
        self.ndf = args.ndf
        self.device = device
        self.d_1 = nn.Sequential(ConvBlock(self.input_nc, self.ndf * 1, stride=2),  # (B, 64, H, W)
                                 ConvBlock(self.ndf * 1, self.ndf * 2, stride=2),   # (B, 128, H/2, W/2)
                                 ConvBlock(self.ndf * 2, self.ndf * 4, stride=2),   # (B, 256, H/4, W/4)
                                 ConvBlock(self.ndf * 4, self.ndf * 8, stride=2),   # (B, 512, H/8, W/8)
                                 nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
                                 )                                                  # (B, 1, H/8, W/8)

        self.d_2 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
                                 ConvBlock(self.input_nc, self.ndf * 1, stride=2),  # (B, 64, H/2, W/2)
                                 ConvBlock(self.ndf * 1, self.ndf * 2, stride=2),   # (B, 128, H/4, W/4)
                                 ConvBlock(self.ndf * 2, self.ndf * 4, stride=2),   # (B, 256, H/8, W/8)
                                 ConvBlock(self.ndf * 4, self.ndf * 8, stride=2),   # (B, 512, H/16, W/16)
                                 nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
                                 )                                                  # (B, 1, H/16, W/16)

        self.d_3 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=4, padding=0, count_include_pad=False),
                                 ConvBlock(self.input_nc, self.ndf * 1, stride=2),  # (B, 64, H/4, W/4)
                                 ConvBlock(self.ndf * 1, self.ndf * 2, stride=2),   # (B, 128, H/8, W/8)
                                 ConvBlock(self.ndf * 2, self.ndf * 4, stride=2),   # (B, 256, H/16, W/16)
                                 ConvBlock(self.ndf * 4, self.ndf * 8, stride=2),   # (B, 512, H/32, W/32)
                                 nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
                                 )                                                  # (B, 1, H/32, W/32)

    def forward(self, img):
        out1 = self.d_1(img)
        out2 = self.d_2(img)
        out3 = self.d_3(img)
        return out1, out2, out3


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, pad=0):
        super(ConvBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                          padding_mode='circular'),
                nn.InstanceNorm2d(self.c_out),
                nn.ReLU(inplace=True))
        elif stride == 2:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((0, 1, 0, 1)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                          padding_mode='circular'),
                nn.InstanceNorm2d(self.c_out),
                nn.ReLU(inplace=True))
        else:
            raise Exception("stride size = 1 or 21")

    def forward(self, maps):
        return self.model(maps)


class Decoder(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=2, pad=1):
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.model = nn.ConvTranspose2d(self.c_in, self.c_out, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, maps):
        return self.model(maps)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.MSELoss()

    def get_target_tensor(self, image, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(image)

    def __call__(self, img, target_is_real):
        img1, img2, img3 = img
        target_tensor1 = self.get_target_tensor(img1, target_is_real)
        target_tensor2 = self.get_target_tensor(img2, target_is_real)
        target_tensor3 = self.get_target_tensor(img3, target_is_real)

        loss = (self.loss(img1, target_tensor1) + self.loss(img2, target_tensor2) + self.loss(img3, target_tensor3)) / 3
        return loss


class DarkChannelLoss(nn.Module):
    def __init__(self, kernel_size=15):
        super(DarkChannelLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 3, H, W), in [-1, 1]
        x = (x + 1.0) / 2.0
        H, W = x.size()[2], x.size()[3]

        # Minimum among three channels
        x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # Minimum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
        x = dark_map.view(-1, 1, H, W)

        # Count Zeros
        #y0 = torch.zeros_like(x)
        #y1 = torch.ones_like(x)
        #x = torch.where(x < 0.1, y0, y1)
        #x = torch.sum(x)
        #x = int(H * W - x)
        return x.clamp(min=0.0, max=0.1)

    def __call__(self, real, fake):
        real_map = self.forward(real)
        fake_map = self.forward(fake)
        return self.loss(real_map, fake_map)


class GradientLoss(nn.Module):
    def __init__(self, kernel_size=3, device="cpu"):
        super(GradientLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)
        self.device = device

    def forward(self, x):
        """
        Sobel Filter
        :param x:
        :return: dh, dv
        """
        # x : (B, 3, H, W)
        x = (x + 1.0) / 2.0

        # Compute a gray-scale image by averaging
        x = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        # weight :
        filter_h = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand(1, 1, 3, 3)
        filter_v = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).expand(1, 1, 3, 3)

        filter_h = filter_h.flip(-1).flip(-2)
        filter_v = filter_v.flip(-1).flip(-2)

        filter_h = filter_h.to(self.device)
        filter_v = filter_v.to(self.device)
        # Convolution
        gradient_h = F.conv2d(x, filter_h)
        gradient_v = F.conv2d(x, filter_v)

        return gradient_h, gradient_v

    def __call__(self, real, fake):
        real_grad_h, real_grad_v = self.forward(real)
        fake_grad_h, fake_grad_v = self.forward(fake)

        diff_hv = torch.abs_(real_grad_h - fake_grad_h) + torch.abs_(real_grad_v - fake_grad_v)

        return torch.mean(diff_hv)

