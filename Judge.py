"""
预测效果评价
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import math
from scipy.signal import convolve2d

# path = '32-128-9'
# path = '64-64-7'
# path = '64-128-5'
# path = '64-128-7-1'
# path = '64-128-7'
# path = '64-128-9-1'
# path = '64-128-9'
# path = '64-128-11'
path = 'Mountain'


def psnr(target, ref, m=255.):
    # 求PSNR
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(m / rmse)


def lapulase(pre_img, ori_img):
    img2gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
    res = cv2.Laplacian(img2gray, cv2.CV_64F, ksize=3)
    score1 = res.var()
    img2gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    res = cv2.Laplacian(img2gray, cv2.CV_64F)
    score2 = res.var()
    return score1, score2


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255.):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))


def ASPSIM(pre, true):
    shape = pre.shape
    r = np.zeros((shape[0], shape[1]))
    p = np.zeros(3)
    t = np.zeros(3)
    p[0] = pre[:, :, 0].mean()
    p[1] = pre[:, :, 1].mean()
    p[2] = pre[:, :, 2].mean()
    t[0] = true[:, :, 0].mean()
    t[1] = true[:, :, 1].mean()
    t[2] = true[:, :, 2].mean()
    for i in range(shape[0]):
        # print(i, '/', shape[0])
        for j in range(shape[1]):
            temp_pre = pre[i][j] - p
            temp_true = true[i][j] - t
            r[i][j] = np.dot(temp_pre, temp_true) / ((np.sum(temp_pre**2)**0.5) * (np.sum(temp_true**2)**0.5))
    asp = r.mean()
    return asp


if __name__ == '__main__':
    # 河南
    """img1 = np.load('./Train/Judge/hr_henan.npy')
    img2 = np.load('./Train/Judge/MERSI_henan.npy')[4:1786, 4:2196]
    img3 = np.load('./Train/Judge/Landsat8_henan.npy')[5:1787, 2:2194]"""
    """img1 = cv2.imread('./Train/Judge/Predict.bmp', cv2.IMREAD_COLOR)
    img2 = cv2.imread('./Train/Judge/lr.bmp', cv2.IMREAD_COLOR)
    img3 = cv2.imread('./Train/Judge/Landsat8_henan.bmp', cv2.IMREAD_COLOR)[5:1787, 2:2194]"""
    """img1 = cv2.imread('./Train/Result/Predict.bmp', cv2.IMREAD_COLOR)
    img2 = cv2.imread('./Train/Test/20191119_0530_MERSI.bmp', cv2.IMREAD_COLOR)"""

    # print(img1.shape, img2.shape)
    # print(lapulase(img1, img2))
    # print('PSNR:\nBicubic:', psnr(img2, img3, m=0.5), '\nDCNN:', psnr(img1, img3, m=0.5))
    # print('Bicubic:', ASPSIM(img2, img3))
    # print('ASPSIM:\nSRCNN:', ASPSIM(img1, img3))
    """temp = 0
    for i in range(3):
        temp += compute_ssim(img2[:, :, i], img3[:, :, i], L=1)
    print('Bicubic:', temp / 3)"""
    """temp = 0
    for i in range(3):
        # print(i, '/ 2')
        temp += compute_ssim(img1[:, :, i], img3[:, :, i], L=1)
    print('SSIM:\nSRCNN:', temp / 3)"""

    # 北京
    img1 = np.load('./Train/Judge/c.npy')
    img2 = np.load('./Train/Judge/b.npy')
    img3 = np.load('./Train/Judge/g.npy')
    img4 = np.load('./Train/Judge/gan.npy')
    img5 = np.load('./Train/Judge/d.npy')
    """img1 = cv2.imread('./Train/Judge/b.bmp', cv2.IMREAD_COLOR)
    img2 = cv2.imread('./Train/Judge/d.bmp', cv2.IMREAD_COLOR)
    img3 = cv2.imread('./Train/Judge/gan.bmp', cv2.IMREAD_COLOR)
    print(lapulase(img2, img1), lapulase(img3, img1))"""

    print('PSNR:\nBicubic:', psnr(img2, img3, m=1), '\nSRCNN:', psnr(img1, img3, m=1),
          '\nSRGAN:', psnr(img4, img3, m=1), '\nDSRCNN:', psnr(img5, img3, m=1))
    print('ASPSIM:\nBicubic:', ASPSIM(img2, img3))
    print('SRCNN:', ASPSIM(img1, img3))
    print('SRGAN:', ASPSIM(img4, img3))
    print('DSRCNN:', ASPSIM(img5, img3))
    temp = 0
    for i in range(3):
        temp += compute_ssim(img2[:, :, i], img3[:, :, i], L=1)
    print('SSIM:\nBicubic:', temp / 3)
    temp = 0
    for i in range(3):
        # print(i, '/ 2')
        temp += compute_ssim(img1[:, :, i], img3[:, :, i], L=1)
    print('SRCNN:', temp / 3)
    temp = 0
    for i in range(3):
        # print(i, '/ 2')
        temp += compute_ssim(img4[:, :, i], img3[:, :, i], L=1)
    print('SRGAN:', temp / 3)
    temp = 0
    for i in range(3):
        # print(i, '/ 2')
        temp += compute_ssim(img5[:, :, i], img3[:, :, i], L=1)
    print('DSRCNN:', temp / 3)

    # import matplotlib.pyplot as plt
    # [b1, g1, r1] = cv2.split(img1)
    # [b2, g2, r2] = cv2.split(img2)
    # [b3, g3, r3] = cv2.split(img3)
    # plt.hist(b3)
    # plt.show()
