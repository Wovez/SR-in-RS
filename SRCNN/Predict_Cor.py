"""
用三通道模型预测高分辨率图像

Todo: 将大幅图像分成块状

v1.0 王剑 2020/2/28
"""

from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.optimizers import SGD, Adam
import numpy as np
import cv2
from Read_Landsat8 import linear_2

date = '20191207_0450'
# date = '20200101_0515'
# date = '20191119_0530'

conv_side = 6


def predict_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict():
    srcnn_model_b = predict_model()
    srcnn_model_b.load_weights('./Train/Model/SRCNN0.h5')
    srcnn_model_g = predict_model()
    srcnn_model_g.load_weights('./Train/Model/SRCNN1.h5')
    srcnn_model_r = predict_model()
    srcnn_model_r.load_weights('./Train/Model/SRCNN2.h5')
    data_path = './Train/Test/' + date + '_MERSI_data_mult.npy'
    data = np.load(data_path)
    shape = data.shape
    print(np.mean(data[:, :, 0]), np.mean(data[:, :, 1]), np.mean(data[:, :, 2]))
    """blue = data[:, :, 0]
    green = data[:, :, 1]
    red = data[:, :, 2]
    blue, green, red = linear_2(blue, green, red, 2)
    rgb = cv2.merge([blue, green, red])
    cv2.imwrite('./Train/Result/MERSI.bmp', rgb)"""

    _r = shape[0] // 1200
    _c = shape[1] // 1500
    if _r * 1200 + conv_side < shape[0]:
        _r += 1
    if _c * 1500 + conv_side < shape[1]:
        _c += 1
    _size = np.array([_r, _c, shape[0], shape[1], shape[2]])
    np.save('./Train/Predict_pre/size', _size)
    for i in range(_r):
        print(i+1, '/', _r)
        for j in range(_c):
            if i == 0:
                if j == 0:
                    _d = data[0:(1200 + conv_side), 0:(1500 + conv_side)]
                else:
                    _d = data[0:(1200 + conv_side), (j * 1500 - conv_side):(j * 1500 + 1500 + conv_side)]
            elif j == 0:
                _d = data[(i * 1200 - conv_side):(i * 1200 + 1200 + conv_side), 0:(1500 + conv_side)]
            else:
                _d = data[(i * 1200 - conv_side):(i * 1200 + 1200 + conv_side),
                          (j * 1500 - conv_side):(j * 1500 + 1500 + conv_side)]
            shape = _d.shape
            f_b = np.zeros((1, shape[0], shape[1], 1), dtype=float)
            f_g = np.zeros((1, shape[0], shape[1], 1), dtype=float)
            f_r = np.zeros((1, shape[0], shape[1], 1), dtype=float)
            f_b[0, :, :, 0] = _d[:, :, 0]
            f_g[0, :, :, 0] = _d[:, :, 1]
            f_r[0, :, :, 0] = _d[:, :, 2]
            pre_b = srcnn_model_b.predict(f_b, batch_size=1)
            pre_g = srcnn_model_g.predict(f_g, batch_size=1)
            pre_r = srcnn_model_r.predict(f_r, batch_size=1)
            blue = pre_b[0, :, :, 0]
            green = pre_g[0, :, :, 0]
            red = pre_r[0, :, :, 0]
            pre_data = cv2.merge([blue, green, red])
            a = str(i*_c+j)
            np.save('./Train/Predict_pre/' + a, pre_data)

    """temp = [data[0:1206, 0:1506], data[1194:2406, 0:1506], data[2394:3333, 0:1506], data[0:1206, 1494:3006],
            data[1194:2406, 1494:3006], data[2394:3333, 1494:3006], data[0:1204, 2996:4534], data[1194:2406, 2994:4534],
            data[2394:3333, 2994:4534]]
    del data
    for i in range(9):
        a = str(i)
        print(a, '/ 8')
        _d = temp[i]
        shape = _d.shape
        f_b = np.zeros((1, shape[0], shape[1], 1), dtype=float)
        f_g = np.zeros((1, shape[0], shape[1], 1), dtype=float)
        f_r = np.zeros((1, shape[0], shape[1], 1), dtype=float)
        f_b[0, :, :, 0] = _d[:, :, 0]
        f_g[0, :, :, 0] = _d[:, :, 1]
        f_r[0, :, :, 0] = _d[:, :, 2]

        pre_b = srcnn_model_b.predict(f_b, batch_size=1)
        pre_g = srcnn_model_g.predict(f_g, batch_size=1)
        pre_r = srcnn_model_r.predict(f_r, batch_size=1)
        blue = pre_b[0, :, :, 0]
        green = pre_g[0, :, :, 0]
        red = pre_r[0, :, :, 0]
        pre_data = cv2.merge([blue, green, red])
        np.save('./Train/0/' + a, pre_data)"""

    """output_path = './Train/Result/Predict.bmp'
    f_b = np.zeros((1, shape[0], shape[1], 1), dtype=float)
    f_g = np.zeros((1, shape[0], shape[1], 1), dtype=float)
    f_r = np.zeros((1, shape[0], shape[1], 1), dtype=float)
    f_b[0, :, :, 0] = data[:, :, 0]
    f_g[0, :, :, 0] = data[:, :, 1]
    f_r[0, :, :, 0] = data[:, :, 2]

    pre_b = srcnn_model_b.predict(f_b, batch_size=1)
    pre_g = srcnn_model_g.predict(f_g, batch_size=1)
    pre_r = srcnn_model_r.predict(f_r, batch_size=1)
    blue = pre_b[0, :, :, 0]
    green = pre_g[0, :, :, 0]
    red = pre_r[0, :, :, 0]

    pre_data = cv2.merge([blue, green, red])
    np.save('./Train/Result/Predict', pre_data)

    blue, green, red = linear_2(blue, green, red, 2)

    output = cv2.merge([blue, green, red])
    print(output.shape)
    cv2.imwrite(output_path, output)"""


def mix():
    _size = np.load('./Train/Predict_pre/size.npy')
    _r = _size[0]
    _c = _size[1]
    shape = [_size[2], _size[3], _size[4]]
    ot = np.zeros((shape[0]-conv_side*2, shape[1]-conv_side*2, shape[2]))
    for i in range(_r):
        for j in range(_c):
            a = str(i*_c+j)
            if i == 0:
                if j == 0:
                    ot[0:(1200 - conv_side), 0:(1500 - conv_side)] = np.load('./Train/Predict_pre/'+a+'.npy')
                else:
                    ot[0:(1200 - conv_side), (j * 1500 - conv_side):(j * 1500 + 1500 - conv_side)] = \
                        np.load('./Train/Predict_pre/'+a+'.npy')
            elif j == 0:
                ot[(i * 1200 - conv_side):(i * 1200 + 1200 - conv_side),
                   0:(1500 - conv_side)] = np.load('./Train/Predict_pre/'+a+'.npy')
            else:
                ot[(i * 1200 - conv_side):(i * 1200 + 1200 - conv_side),
                   (j * 1500 - conv_side):(j * 1500 + 1500 - conv_side)] = np.load('./Train/Predict_pre/'+a+'.npy')
    return ot


if __name__ == '__main__':
    predict()
    temp = mix()
    blue = temp[:, :, 0]
    green = temp[:, :, 1]
    red = temp[:, :, 2]
    rgb = cv2.merge([blue, green, red])
    np.save('./Train/Result/Predict', rgb)
    print(np.mean(blue), np.mean(green), np.mean(red))
    blue, green, red = linear_2(blue, green, red, 2)
    rgb = cv2.merge([blue, green, red])
    cv2.imwrite('./Train/Result/Predict.bmp', rgb)
