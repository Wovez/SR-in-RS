import numpy as np
import cv2
import matplotlib.pyplot as plt

# date = '20200101_0515'
date = '20191207_0450'
# date = '20191119_0530'


def affine_transform(_img1, _img2):
    # 仿射变换
    b1, g1, r1 = cv2.split(_img1)
    b2, g2, r2 = cv2.split(_img2)
    _img1 = cv2.merge([r1, g1, b1])
    _img2 = cv2.merge([r2, g2, b2])
    plt.imshow(_img2, cmap=plt.get_cmap("gray"))
    _pts2 = plt.ginput(3)
    plt.close()
    plt.imshow(_img1, cmap=plt.get_cmap("gray"))
    _pts1 = plt.ginput(3)
    _pts1 = np.array(_pts1, dtype="float32")
    _pts2 = np.array(_pts2, dtype="float32")
    _M = cv2.getAffineTransform(_pts2, _pts1)
    # _M = cv2.getPerspectiveTransform(_pts2, _pts1)
    np.save('./Data/MERSI/Final/'+date+'_AffineMat', _M)


def test(_img1, _img2):
    _shape = _img2.shape
    _M = np.load('./Data/MERSI/Final/' + date + '_AffineMat.npy')
    _result = cv2.warpAffine(_img1, _M, (_shape[1], _shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # _result = cv2.warpPerspective(_img1, M, (_shape[1], _shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imwrite('./Data/img1.bmp', _img1)
    cv2.imwrite('./Data/img2.bmp', _img2)
    cv2.imwrite('./Data/result.bmp', _result)


def affine_cor():
    _M = np.load('./Data/MERSI/Final/' + date + '_AffineMat.npy')

    _img1 = cv2.imread('./Data/MERSI/Final/' + date + '_Landsat8.bmp', cv2.IMREAD_COLOR)
    _img2 = cv2.imread('./Data/MERSI/Final/' + date + '_MERSI.bmp', cv2.IMREAD_COLOR)
    _shape = _img2.shape
    _img1 = cv2.resize(_img1, (_shape[1] * 4, _shape[0] * 4), cv2.INTER_CUBIC)
    _img2 = cv2.resize(_img2, (_shape[1] * 4, _shape[0] * 4), cv2.INTER_CUBIC)
    _result = cv2.warpAffine(_img1, _M, (_shape[1] * 4, _shape[0] * 4), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imwrite('./Data/MERSI/Affine/'+date+'_MERSI.bmp', _img2)
    cv2.imwrite('./Data/MERSI/Affine/'+date+'_Landsat8.bmp', _result)

    _img1 = np.load('./Data/MERSI/Final/' + date + '_Landsat8_data.npy')
    _img2 = np.load('./Data/MERSI/Final/' + date + '_MERSI_data.npy')
    _shape = _img2.shape
    _img1 = cv2.resize(_img1, (_shape[1] * 4, _shape[0] * 4), cv2.INTER_CUBIC)
    _img2 = cv2.resize(_img2, (_shape[1] * 4, _shape[0] * 4), cv2.INTER_CUBIC)
    _result = cv2.warpAffine(_img1, _M, (_shape[1] * 4, _shape[0] * 4), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    np.save('./Data/MERSI/Affine/' + date + '_MERSI_data', _img2)
    np.save('./Data/MERSI/Affine/' + date + '_Landsat8_data', _result)


def cut():
    _img1 = cv2.imread('./Data/MERSI/Affine/' + date + '_Landsat8.bmp', cv2.IMREAD_COLOR)
    _img2 = cv2.imread('./Data/MERSI/Affine/' + date + '_MERSI.bmp', cv2.IMREAD_COLOR)
    """_img1 = _img1[10:1800, 0:2200]   # 河南
    _img2 = _img2[10:1800, 0:2200]"""
    _img1 = _img1[20:3200, 20:2770]   # 北京
    _img2 = _img2[20:3200, 20:2770]
    cv2.imwrite('./Data/MERSI/Affine/' + date + '_MERSI.bmp', _img2)
    cv2.imwrite('./Data/MERSI/Affine/' + date + '_Landsat8.bmp', _img1)

    _img1 = np.load('./Data/MERSI/Affine/' + date + '_Landsat8_data.npy')
    _img2 = np.load('./Data/MERSI/Affine/' + date + '_MERSI_data.npy')
    """_img1 = _img1[10:1800, 0:2200]   # 河南
    _img2 = _img2[10:1800, 0:2200]"""
    _img1 = _img1[20:3200, 20:2770]   # 北京
    _img2 = _img2[20:3200, 20:2770]
    np.save('./Data/MERSI/Affine/' + date + '_MERSI_data', _img2)
    np.save('./Data/MERSI/Affine/' + date + '_Landsat8_data', _img1)


if __name__ == '__main__':
    # 手动选出仿射变换矩阵M
    """img1 = cv2.imread('./Data/MERSI/Final/'+date+'_Landsat8.bmp', cv2.IMREAD_COLOR)
    img2 = cv2.imread('./Data/MERSI/Final/'+date+'_MERSI.bmp', cv2.IMREAD_COLOR)
    shape = img2.shape
    img1 = cv2.resize(img1, (shape[1] * 4, shape[0] * 4), cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (shape[1] * 4, shape[0] * 4), cv2.INTER_CUBIC)
    affine_transform(img1, img2)
    test(img1, img2)"""

    # 最终校正
    affine_cor()

    # 裁剪多余部分
    cut()
