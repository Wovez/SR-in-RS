import cv2
import os
import numpy as np
from Read_MERSI import linear_2

dataset_name = '20191207_0450'


def pre():
    lr_data = np.load('./Datasets/' + dataset_name + '/origin/MERSI_data.npy')
    hr_data = np.load('./Datasets/' + dataset_name + '/origin/Landsat8_data.npy')
    lr_img = cv2.imread('./Datasets/' + dataset_name + '/origin/MERSI.bmp', cv2.IMREAD_COLOR)
    hr_img = cv2.imread('./Datasets/' + dataset_name + '/origin/Landsat8.bmp', cv2.IMREAD_COLOR)

    shape = lr_img.shape
    hr_data = cv2.resize(hr_data, (shape[1] * 4, shape[0] * 4))
    hr_img = cv2.resize(hr_img, (shape[1] * 4, shape[0] * 4))

    lr_img = lr_img[5:500, 0:670][:]
    lr_data = lr_data[5:500, 0:670][:]
    hr_img = hr_img[20:2000, 0:2680][:]
    hr_data = hr_data[20:2000, 0:2680][:]

    lr_data[lr_data[:] > 0.5] = 0.5
    lr_data[lr_data[:] < 0] = 0
    hr_data[hr_data[:] > 0.5] = 0.5
    hr_data[hr_data[:] < 0] = 0
    print(lr_img.shape, lr_data.shape, hr_img.shape, hr_data.shape)

    np.save('./Datasets/' + dataset_name + '/origin/MERSI_data.npy', lr_data)
    np.save('./Datasets/' + dataset_name + '/origin/Landsat8_data.npy', hr_data)
    cv2.imwrite('./Datasets/' + dataset_name + '/origin/MERSI.bmp', lr_img)
    cv2.imwrite('./Datasets/' + dataset_name + '/origin/Landsat8.bmp', hr_img)


def cut(img_res=(96, 96)):
    # 裁剪为子图
    lr_data = np.load('./Datasets/' + dataset_name + '/origin/MERSI_data.npy')
    hr_data = np.load('./Datasets/' + dataset_name + '/origin/Landsat8_data.npy')
    _shape = lr_data.shape

    # 确定裁剪数量
    _row = (_shape[0] - int(img_res[0]/2)) // int(img_res[0]/2)
    _temp = (img_res[0] * (_row + 1) - _shape[0]) // _row + 1
    _row = (_shape[0] - _temp) // (img_res[0] - _temp)
    _row_cut = img_res[0] - _temp
    _col = (_shape[1] - int(img_res[1]/2)) // int(img_res[1]/2)
    _temp = (img_res[1] * (_col + 1) - _shape[1]) // _col + 1
    _col = (_shape[1] - _temp) // (img_res[1] - _temp)
    _col_cut = img_res[1] - _temp

    for i in range(_row):
        for j in range(_col):
            temp = str(i * _row + j)
            lr_cropped = lr_data[(i * _row_cut):(i * _row_cut + img_res[0]), (j * _col_cut):(j * _col_cut + img_res[1])]
            hr_cropped = hr_data[(i * _row_cut * 4):(i * _row_cut * 4 + img_res[0] * 4),
                                 (j * _col_cut * 4):(j * _col_cut * 4 + img_res[1] * 4)]
            np.save('./Datasets/' + dataset_name + '/init/{}_MERSI_data'.format(temp), lr_cropped)
            np.save('./Datasets/' + dataset_name + '/init/{}_Landsat8_data'.format(temp), hr_cropped)


def prepare_data():
    path = './Datasets/' + dataset_name + '/init/'
    outpath = './Datasets/' + dataset_name + '/train/'
    data_names = os.listdir(path)
    data_names = sorted(data_names)
    data_nums = data_names.__len__()

    imgs_hr = []
    imgs_lr = []

    for i in range(int(data_nums/2)):
        _hr_path = path + data_names[i * 2]
        _lr_path = path + data_names[i * 2 + 1]
        _hr_data = np.load(_hr_path)
        _lr_data = np.load(_lr_path)

        imgs_hr.append(_hr_data)
        imgs_lr.append(_lr_data)

    imgs_hr = np.array(imgs_hr) / 0.25 - 1.
    imgs_lr = np.array(imgs_lr) / 0.25 - 1.

    np.save(outpath + 'imgs_hr', imgs_hr)
    np.save(outpath + 'imgs_lr', imgs_lr)


if __name__ == '__main__':
    # pre()
    cut(img_res=(64, 64))
    prepare_data()
