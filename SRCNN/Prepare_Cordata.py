"""
功能：
    生成从MERSI直接映射为Landsat8分辨率的训练集

v1.0 王剑 2020/02/27
"""
import cv2
import numpy as np
import h5py
import os

# date = '20200101_0515'
date = '20191207_0450'
CV_path = './Train/CV/'
Train_path = './Train/Set0/'
Patch_size = 64         # 图像块大小
Random_crop = 64        # 单个图象划分图像块个数
label_size = 52         # 目标图像块大小
conv_side = 6           # 考虑边缘效应(由滤波器决定)
BLOCK_STEP = 32
BLOCK_SIZE = 64


def Cut():
    # 裁剪为子图像
    hr_img = cv2.imread('./Data/MERSI/Affine/' + date + '_Landsat8.bmp', cv2.IMREAD_COLOR)
    hr_data = np.load('./Data/MERSI/Affine/' + date + '_Landsat8_data.npy')
    lr_img = cv2.imread('./Data/MERSI/Affine/' + date + '_MERSI.bmp', cv2.IMREAD_COLOR)
    lr_data = np.load('./Data/MERSI/Affine/' + date + '_MERSI_data.npy')
    """if scale == '4':
        lr_img = cv2.imread('./Data/MERSI/Affine/' + date + '_MERSI.bmp', cv2.IMREAD_COLOR)
        lr_data = np.load('./Data/MERSI/Affine/' + date + '_MERSI_data.npy')
        _shape_hr = hr_img.shape
        _shape_lr = [int(_shape_hr[0] / 4), int(_shape_hr[1] / 4), 3]
        print(_shape_hr, _shape_lr)
        hr_img = cv2.resize(hr_img, (_shape_lr[1], _shape_lr[0]))
        lr_img = cv2.resize(lr_img, (_shape_lr[1], _shape_lr[0]))
        hr_data = cv2.resize(hr_data, (_shape_lr[1], _shape_lr[0]))
        lr_data = cv2.resize(lr_data, (_shape_lr[1], _shape_lr[0]))
    elif scale == '2':
        lr_img = cv2.imread('./Train/4x/Result/MERSI.bmp', cv2.IMREAD_COLOR)
    cv2.imwrite('./Train/' + scale + 'x/Result/Landsat8.bmp', hr_img)
    cv2.imwrite('./Train/' + scale + 'x/Result/MERSI.bmp', lr_img)"""
    _shape = hr_img.shape

    # 确定裁剪数量
    _row = (_shape[0] - 32) // 224
    _temp = (256 * (_row + 1) - _shape[0]) // _row + 1
    _row = (_shape[0] - _temp) // (256 - _temp)
    _row_cut = 256 - _temp
    _col = (_shape[1] - 32) // 224
    _temp = (256 * (_col + 1) - _shape[1]) // _col + 1
    _col = (_shape[1] - _temp) // (256 - _temp)
    _col_cut = 256 - _temp

    # 图像格式和数据格式分别裁剪
    for i in range(_row):
        for j in range(_col):
            temp = str(i * _col + j)
            lr_cropped = lr_img[(i * _row_cut):(i * _row_cut + 256), (j * _col_cut):(j * _col_cut + 256)]
            hr_cropped = hr_img[(i * _row_cut):(i * _row_cut + 256), (j * _col_cut):(j * _col_cut + 256)]
            cv2.imwrite('./Data/Set_Cor/{}_MERSI.bmp'.format(temp), lr_cropped)
            cv2.imwrite('./Data/Set_Cor/{}_Landsat.bmp'.format(temp), hr_cropped)
            lr_cropped = lr_data[(i * _row_cut):(i * _row_cut + 256), (j * _col_cut):(j * _col_cut + 256)]
            hr_cropped = hr_data[(i * _row_cut):(i * _row_cut + 256), (j * _col_cut):(j * _col_cut + 256)]
            np.save('./Data/Set_Cor_data/{}_MERSI_data'.format(temp), lr_cropped)
            np.save('./Data/Set_Cor_data/{}_Landsat8_data'.format(temp), hr_cropped)


def prepare_CV(_b):
    # 准备交叉验证样本
    _data_names = os.listdir(CV_path)        # 返回指定（交叉验证数据）文件夹包含的文件（数据）名字的列表
    _data_names = sorted(_data_names)         # 对图片名称进行排序
    _data_nums = _data_names.__len__()        # 统计交叉验证图像个数

    _data = np.zeros((int(_data_nums/2) * Random_crop, 1, Patch_size, Patch_size), dtype=np.double)
    _label = np.zeros((int(_data_nums/2) * Random_crop, 1, label_size, label_size), dtype=np.double)

    for i in range(int(_data_nums/2)):
        _hr_path = CV_path + _data_names[i*2]
        _lr_path = CV_path + _data_names[i*2+1]
        _hr_data = np.load(_hr_path)[:, :, _b]
        _lr_data = np.load(_lr_path)[:, :, _b]
        shape = _hr_data.shape

        Points_x = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_crop)
        Points_y = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_crop)

        for j in range(Random_crop):
            _hr_patch = _hr_data[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            _lr_patch = _lr_data[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

            _hr_patch = _hr_patch.astype(np.float)
            _lr_patch = _lr_patch.astype(np.float)

            _data[i * Random_crop + j, 0, :] = _lr_patch
            _label[i * Random_crop + j, 0, :] = _hr_patch[conv_side: -conv_side, conv_side: -conv_side]
    print(_data.shape)
    return _data, _label


def prepare_train(_b):
    # 准备训练样本
    _data_names = os.listdir(Train_path)
    _data_names = sorted(_data_names)
    _data_nums = _data_names.__len__()

    _data = []
    _label = []

    for i in range(int(_data_nums/2)):
        _hr_path = Train_path + _data_names[i*2]
        _lr_path = Train_path + _data_names[i*2+1]
        _hr_data = np.load(_hr_path)[:, :, _b]
        _lr_data = np.load(_lr_path)[:, :, _b]
        _shape = _hr_data.shape

        _width_num = int((_shape[0] - (BLOCK_SIZE - BLOCK_STEP)) / BLOCK_STEP)
        _height_num = int((_shape[1] - (BLOCK_SIZE - BLOCK_STEP)) / BLOCK_STEP)

        for k in range(_width_num):
            for j in range(_height_num):
                x = k * BLOCK_STEP
                y = j * BLOCK_STEP
                _hr_patch = _hr_data[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                _lr_patch = _lr_data[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                _lr_patch = _lr_patch.astype(float)
                _hr_patch = _hr_patch.astype(float)

                _lr = np.zeros((1, Patch_size, Patch_size), dtype=np.double)
                _hr = np.zeros((1, label_size, label_size), dtype=np.double)

                _lr[0, :] = _lr_patch
                _hr[0, :] = _hr_patch[conv_side: -conv_side, conv_side: -conv_side]

                _data.append(_lr)
                _label.append(_hr)

    _data = np.array(_data, dtype=float)
    _label = np.array(_label, dtype=float)
    print(_data.shape)
    return _data, _label


def write_hdf5(data, labels, output_filename):
    # 保存训练、测试的数据和标签为h5文件

    x = data.astype(np.float32)
    y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)


if __name__ == '__main__':
    # Cut()
    for band in range(3):
        temp = str(band)
        data, label = prepare_CV(band)
        write_hdf5(data, label, './Train/Model/CV' + temp + '.h5')
        data, label = prepare_train(band)
        write_hdf5(data, label, './Train/Model/Train' + temp + '.h5')
