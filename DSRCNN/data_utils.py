import numpy as np
import random

dataset_name = '20191207_0450'
conv_side = 6


def cut(img_res=(96, 96)):
    outpath = './Datasets/' + dataset_name + '/train/'

    # 裁剪为子图
    lr_data = np.load('./Datasets/' + dataset_name + '/origin/MERSI_data.npy')
    hr_data = np.load('./Datasets/' + dataset_name + '/origin/Landsat8_data.npy')
    hr_data[:, :, 0] = hr_data[:, :, 0] + round(np.mean(lr_data[:, :, 0]) - np.mean(hr_data[:, :, 0]), 4)
    hr_data[:, :, 1] = hr_data[:, :, 1] + round(np.mean(lr_data[:, :, 1]) - np.mean(hr_data[:, :, 1]), 4)
    hr_data[:, :, 2] = hr_data[:, :, 2] + round(np.mean(lr_data[:, :, 2]) - np.mean(hr_data[:, :, 2]), 4)
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

    imgs_hr = []
    imgs_lr = []
    val_data = []
    val_label = []

    for i in range(_row):
        for j in range(_col):
            lr_cropped = lr_data[(i * _row_cut):(i * _row_cut + img_res[0]), (j * _col_cut):(j * _col_cut + img_res[1])]
            hr_cropped = hr_data[(i * _row_cut):(i * _row_cut + img_res[0]),
                                 (j * _col_cut):(j * _col_cut + img_res[1])]

            if random.random() > 0.8:
                val_label.append(hr_cropped[conv_side: -conv_side, conv_side: -conv_side])
                val_data.append(lr_cropped)
            else:
                imgs_hr.append(hr_cropped[conv_side: -conv_side, conv_side: -conv_side])
                imgs_lr.append(lr_cropped)

    imgs_hr = np.array(imgs_hr)
    imgs_lr = np.array(imgs_lr)
    val_data = np.array(val_data)
    val_label = np.array(val_label)

    np.save(outpath + 'imgs_hr', imgs_hr)
    np.save(outpath + 'imgs_lr', imgs_lr)
    np.save(outpath + 'val_data', val_data)
    np.save(outpath + 'val_label', val_label)

    print(imgs_hr.shape, val_label.shape)


if __name__ == '__main__':
    cut(img_res=(64, 64))
