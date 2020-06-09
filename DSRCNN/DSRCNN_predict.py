import cv2
import numpy as np
from Read_MERSI import linear_2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, Input
from tensorflow.keras.layers import Conv2D, UpSampling2D
import time

dataset_name = '20191207_0450'
conv_side = 6


def Generator():
    # 生成器
    def residual(input_layer, filters):
        r = Conv2D(filters, kernel_size=3, strides=1, padding='same')(input_layer)
        r = Activation('relu')(r)
        r = Conv2D(filters, kernel_size=3, strides=1, padding='same')(r)
        r = Add()([r, input_layer])
        return r

    img_lr = Input(shape=[None, None, 1])
    g1 = Conv2D(256, kernel_size=9, strides=1, padding='valid')(img_lr)
    g1 = Activation('relu')(g1)

    g2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(g1)
    g2 = residual(g2, 64)
    for _ in range(15):
        g2 = residual(g2, 64)

    gen_hr = Conv2D(1, kernel_size=5, strides=1, padding='valid', activation='tanh')(g2)
    return Model(img_lr, gen_hr)


def predict(lr_data, batch=(600, 600)):
    shape = lr_data.shape
    row = shape[0] // batch[0]
    if shape[0] % (row * batch[0]) != 0:
        row += 1
    col = shape[1] // batch[1]
    if shape[1] % (col * batch[1]) != 0:
        col += 1
    # output = np.zeros((shape[0] * 4, shape[1] * 4, shape[2]))
    output = np.zeros((shape[0] - conv_side * 2, shape[1] - conv_side * 2, shape[2]))
    for b in range(shape[2]):
        weight = './Model/' + dataset_name + '/DCNN_' + str(b) + '.h5'
        gen = Generator()
        gen.load_weights(weight)
        for i in range(row):
            print(i, '/', row-1)
            for j in range(col):
                if i == 0:
                    if j == 0:
                        _shape = lr_data[0:(batch[0] + conv_side),
                                         0:(batch[1] + conv_side), b].shape
                        lr_temp = np.zeros((1, _shape[0], _shape[1], 1))
                        lr_temp[0, :, :, 0] = lr_data[0:(batch[0] + conv_side),
                                                      0:(batch[1] + conv_side), b]
                        hr_temp = gen.predict(lr_temp)
                        output[0:(batch[0] - conv_side),
                               0:(batch[1] - conv_side), b] = hr_temp[0, :, :, 0]
                    else:
                        _shape = lr_data[0:(batch[0] + conv_side),
                                         (j * batch[1] - conv_side):((j + 1) * batch[1] + conv_side), b].shape
                        lr_temp = np.zeros((1, _shape[0], _shape[1], 1))
                        lr_temp[0, :, :, 0] = lr_data[0:(batch[0] + conv_side),
                                                      (j * batch[1] - conv_side):((j + 1) * batch[1] + conv_side), b]
                        hr_temp = gen.predict(lr_temp)
                        output[0:(batch[0] - conv_side),
                               (j * batch[1] - conv_side):((j + 1) * batch[1] - conv_side), b] = hr_temp[0, :, :, 0]
                elif j == 0:
                    _shape = lr_data[(i * batch[0] - conv_side):((i + 1) * batch[0] + conv_side),
                                     0:(batch[1] + conv_side), b].shape
                    lr_temp = np.zeros((1, _shape[0], _shape[1], 1))
                    lr_temp[0, :, :, 0] = lr_data[(i * batch[0] - conv_side):((i + 1) * batch[0] + conv_side),
                                                  0:(batch[1] + conv_side), b]
                    hr_temp = gen.predict(lr_temp)
                    output[(i * batch[0] - conv_side):((i + 1) * batch[0] - conv_side),
                           0:(batch[1] - conv_side), b] = hr_temp[0, :, :, 0]
                else:
                    _shape = lr_data[(i * batch[0] - conv_side):((i + 1) * batch[0] + conv_side),
                                     (j * batch[1] - conv_side):((j + 1) * batch[1] + conv_side), b].shape
                    lr_temp = np.zeros((1, _shape[0], _shape[1], 1))
                    lr_temp[0, :, :, 0] = lr_data[(i * batch[0] - conv_side):((i + 1) * batch[0] + conv_side),
                                                  (j * batch[1] - conv_side):((j + 1) * batch[1] + conv_side), b]
                    hr_temp = gen.predict(lr_temp)
                    output[(i * batch[0] - conv_side):((i + 1) * batch[0] - conv_side),
                           (j * batch[1] - conv_side):((j + 1) * batch[1] - conv_side), b] = hr_temp[0, :, :, 0]
                """_shape = lr_data[i * batch[0]:(i * batch[0] + batch[0]),
                                 j * batch[1]:(j * batch[1] + batch[1]), b].shape
                lr_temp = np.zeros((1, _shape[0], _shape[1], 1))
                lr_temp[0, :, :, 0] = lr_data[i * batch[0]:(i * batch[0] + batch[0]),
                                              j * batch[1]:(j * batch[1] + batch[1]), b]
                hr_temp = gen.predict(lr_temp)
                output[i * batch[0]:(i * batch[0] + batch[0]),
                       j * batch[1]:(j * batch[1] + batch[1]), b] = hr_temp[0, :, :, 0]"""
    return output


if __name__ == '__main__':
    lr = np.load('./Datasets/' + dataset_name + '/origin/MERSI_data_s.npy')
    lr = np.array(lr)
    start = time.time()
    hr = predict(lr)
    end = time.time()
    print('Predict Time:', end - start)
    np.save('./Result/' + dataset_name + '/hr', hr)
    blue = hr[:, :, 0]
    green = hr[:, :, 1]
    red = hr[:, :, 2]
    print(np.mean(blue), np.mean(green), np.mean(red))
    blue, green, red = linear_2(blue, green, red)
    rgb = cv2.merge([blue, green, red])
    cv2.imwrite('./Result/' + dataset_name + '/hr.bmp', rgb)
    blue = lr[:, :, 0]
    green = lr[:, :, 1]
    red = lr[:, :, 2]
    print(np.mean(blue), np.mean(green), np.mean(red))
    blue, green, red = linear_2(blue, green, red)
    rgb = cv2.merge([blue, green, red])
    cv2.imwrite('./Result/' + dataset_name + '/lr.bmp', rgb)
