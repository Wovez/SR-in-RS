import cv2
import numpy as np
from Read_MERSI import linear_2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Input
from tensorflow.keras.layers import Conv2D, UpSampling2D

dataset_name = '20191207_0450'
epoch = 'epoch_final_'


def Generator():
    # 生成器
    def residual(input_layer, filters):
        r = Conv2D(filters, kernel_size=3, strides=1, padding='same')(input_layer)
        r = BatchNormalization(momentum=0.8)(r)
        r = Activation('relu')(r)
        r = Conv2D(filters, kernel_size=3, strides=1, padding='same')(r)
        r = BatchNormalization(momentum=0.8)(r)
        r = Add()([r, input_layer])
        return r

    def deconv2d(input_layer):
        d = UpSampling2D(size=2)(input_layer)
        d = Conv2D(256, kernel_size=3, strides=1, padding='same')(d)
        d = Activation('relu')(d)
        return d

    img_lr = Input(shape=[None, None, 1])
    g1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(img_lr)
    g1 = Activation('relu')(g1)

    g2 = residual(g1, 64)
    for _ in range(15):
        g2 = residual(g2, 64)
    g2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(g2)
    g2 = BatchNormalization(momentum=0.8)(g2)
    g2 = Add()([g2, g1])

    g3 = deconv2d(g2)
    g3 = deconv2d(g3)
    gen_hr = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='tanh')(g3)
    return Model(img_lr, gen_hr)


def predict(lr_data, batch=(150, 150)):
    hr_batch = (batch[0] * 4, batch[1] * 4)
    shape = lr_data.shape
    row = shape[0] // batch[0]
    if shape[0] % (row * batch[0]) != 0:
        row += 1
    col = shape[1] // batch[1]
    if shape[1] % (col * batch[1]) != 0:
        col += 1
    output = np.zeros((shape[0] * 4, shape[1] * 4, shape[2]))
    for b in range(shape[2]):
        weight = './Model/' + dataset_name + '/gen_' + epoch + str(b) + '.h5'
        gen = Generator()
        gen.load_weights(weight)
        for i in range(row):
            print(i, '/', row-1)
            for j in range(col):
                _shape = lr_data[i * batch[0]:(i * batch[0] + batch[0]),
                                 j * batch[1]:(j * batch[1] + batch[1]), b].shape
                lr_temp = np.zeros((1, _shape[0], _shape[1], 1))
                lr_temp[0, :, :, 0] = lr_data[i * batch[0]:(i * batch[0] + batch[0]),
                                              j * batch[1]:(j * batch[1] + batch[1]), b]
                hr_temp = gen.predict(lr_temp)
                output[i * hr_batch[0]:(i * hr_batch[0] + hr_batch[0]),
                       j * hr_batch[1]:(j * hr_batch[1] + hr_batch[1]), b] = hr_temp[0, :, :, 0]
    return output


if __name__ == '__main__':
    lr = np.load('./Datasets/' + dataset_name + '/origin/MERSI_data.npy')
    lr = np.array(lr) / 0.25 - 1
    hr = predict(lr)
    hr = (hr + 1) * 0.25
    np.save('./Result/' + dataset_name + '/hr_' + epoch, hr)
    blue = hr[:, :, 0]
    green = hr[:, :, 1]
    red = hr[:, :, 2]
    print(np.mean(blue), np.mean(green), np.mean(red))
    blue, green, red = linear_2(blue, green, red)
    rgb = cv2.merge([blue, green, red])
    cv2.imwrite('./Result/' + dataset_name + '/hr_' + epoch + '.bmp', rgb)
    blue = lr[:, :, 0] * 0.25 + 0.25
    green = lr[:, :, 1] * 0.25 + 0.25
    red = lr[:, :, 2] * 0.25 + 0.25
    print(np.mean(blue), np.mean(green), np.mean(red))
    blue, green, red = linear_2(blue, green, red)
    rgb = cv2.merge([blue, green, red])
    cv2.imwrite('./Result/' + dataset_name + '/lr.bmp', rgb)
