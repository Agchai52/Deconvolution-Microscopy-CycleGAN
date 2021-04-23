import numpy as np
import cv2
import os
import argparse
from skimage.exposure import rescale_intensity
from scipy.stats import multivariate_normal
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser('Geneate dataset BlurMicroscopy')
parser.add_argument('--output_fold', dest='output_fold', help='output directory', type=str, default='./dataset/')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=3000)
parser.add_argument('--image_size', dest='image_size', help='image_size = H = W', type=int, default=256)
parser.add_argument('--std_r', dest='std_r', help='std of additive Gaussion white noise', type=float, default=5.)
parser.add_argument('--phase', dest='phase', help='test or train', type=str, default='train')
args = parser.parse_args()


def Gaussian_2D(m=0, sigma=1.):
    """
    :param M: sample num
    :param m: mean
    :param sigma: std
    :return: Gaussian distribution
    """
    mean = np.zeros(2) + m
    cov = np.eye(2) * sigma
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    return Gaussian


def generate_bean(bean_size=10, sigma=0.2, M=50, is_plot=False):
    """
    :param bean_size: bean_size
    :param sigma: std
    :param M: sample num
    :param is_plot: bool plot images
    :return: image of a bean
    """
    Gaussian = Gaussian_2D(m=0, sigma=sigma)
    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, M), np.linspace(-0.5, 0.5, M))
    d = np.dstack([X, Y])
    Z = np.zeros((M, M))
    for i in range(len(d)):
        for j in range(len(d[0])):
            x, y = d[i][j]
            if x ** 2 + y ** 2 <= 0.5*0.5-0.01:
                Z[i][j] = Gaussian.pdf((x, y))

    Z = Z.reshape(M, M)
    max_Z = np.max(Z)
    img_Z = np.uint8(np.asarray(Z)/max_Z*255)
    indensity = np.random.uniform(low=0.5, high=1.0)
    img_Z = indensity * img_Z

    if bean_size < M:
        bean = cv2.resize(img_Z, (bean_size, bean_size), interpolation=cv2.INTER_CUBIC)
    else:
        bean = img_Z

    if is_plot:
        cv2.imwrite("bean_size10.png", bean)
        cv2.imwrite("bean_size50.png", img_Z)

        plt.figure(0)
        plt.imshow(img_Z, cmap='gray', vmin=0, vmax=255)
        plt.figure(1)
        plt.imshow(bean, cmap='gray', vmin=0, vmax=255)

        plt.show()
    return bean


def generate_sharp_img(image_size=256, bean_size=10, bean_min=3, bean_max=10, is_plot=False):
    """
    Generate a sharp image with beans
    :param image_size: image_H = image_W = image_size
    :param bean_size: diameter
    :param bean_min: min num of beans
    :param bean_max: man num of beans
    :param is_plot: bool plot images
    :return: a sharp image with beans
    """
    bean = generate_bean(bean_size=bean_size, sigma=0.2, M=50)
    bean_num = np.random.randint(low=bean_min, high=bean_max)
    bean_locs = np.random.randint(low=bean_size/2+1, high=image_size-bean_size/2-1, size=(bean_num, 2))

    background = np.zeros((image_size, image_size))

    for bean_loc in bean_locs:
        bean_loc_x, bean_loc_y = bean_loc
        left = bean_loc_y - bean_size/2 if 0 < bean_loc_y - bean_size/2 < image_size else 0
        right = bean_loc_y + bean_size/2 if 0 < bean_loc_x - bean_size/2 < image_size else image_size
        up = bean_loc_x - bean_size/2 if 0 < bean_loc_x - bean_size/2 < image_size else 0
        down = bean_loc_x + bean_size/2 if 0 < bean_loc_x - bean_size/2 < image_size else image_size

        background[up:down, left:right] += bean

    if is_plot:
        # cv2.imwrite("sample_img.png", background)
        plt.figure()
        plt.imshow(background, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return background


def convolve(image, kernel):
    """
    2D convolution
    :param image:
    :param kernel:
    :return: convolved image
    """
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
        # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output*255).astype("uint8")
    return output


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


def get_kernel(is_plot=False):
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
    crop_Z = img_Z[crop_size:M-crop_size, crop_size:M-crop_size]
    kernel = crop_Z / np.float(np.sum(crop_Z))
    if is_plot:
        print(crop_Z.shape)
        print(crop_Z)
        # psf = cv2.imread("psf.png", 0)
        # plt.figure()
        # plt.imshow(psf, cmap='gray', vmin=0, vmax=255)
        plt.figure()
        plt.imshow(img_Z, cmap='gray', vmin=0, vmax=255)
        plt.figure()
        plt.imshow(crop_Z, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return kernel


def generate_dataset(name_folder, num_imgs, image_size=256, std_r=5, bean_size=10, bean_min=3, bean_max=10):
    is_plot = False
    kernel = get_kernel()

    for i in range(num_imgs):
        name_prefix = '%04d' % (i+1)
        name_blur = name_folder + name_prefix + "_blur.png"
        name_sharp = name_folder + name_prefix + "_sharp.png"

        sharp = generate_sharp_img(image_size=image_size, bean_size=bean_size,
                                   bean_min=bean_min, bean_max=bean_max, is_plot=False)
        blurry = cv2.filter2D(sharp, -1, kernel)  # convolve(sharp, kernel)
        noise_img = np.random.normal(loc=0, scale=std_r, size=blurry.shape)
        blurry_noisy = blurry + noise_img

        cv2.imwrite(name_sharp, sharp)
        cv2.imwrite(name_blur, blurry_noisy)

        if is_plot:
            # cv2.imwrite("sharp_sample.png", sharp)
            # cv2.imwrite("blurry_sample.png", blurry)
            # cv2.imwrite("blurry_noisy_sample.png", blurry_noisy)

            plt.figure()
            plt.imshow(sharp, cmap='gray', vmin=0, vmax=255)
            plt.figure()
            plt.imshow(blurry, cmap='gray', vmin=0, vmax=255)
            plt.figure()
            plt.imshow(blurry_noisy, cmap='gray', vmin=0, vmax=255)
            plt.show()
    return


if __name__ == "__main__":
    """
    Example:
        python generate_dataset.py --phase train --num_imgs 5
    """
    args = parser.parse_args()

    img_output_fold = args.output_fold + args.phase + '/'  # Output Folder

    if not os.path.isdir(img_output_fold):
        os.makedirs(img_output_fold)

    generate_dataset(img_output_fold, args.num_imgs, image_size=args.image_size, std_r=args.std_r)

    # bean = generate_bean(bean_size=3, sigma=0.2, M=50)
    # cv2.imwrite("bean_size3.png", bean)
