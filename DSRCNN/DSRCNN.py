"""
训练DCNN模型

Todo: first_layer_kernel_size

王剑
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Add, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow


def custom_loss(y_true, y_pred):
    loss_function = 0.8 * K.mean(K.square(y_pred - y_true), axis=-1) + 0.2 * K.mean(K.abs(y_pred - y_true), axis=-1)
    return loss_function


def ssim_loss(y_true, y_pred):
    loss_function = 1 - tensorflow.image.ssim(y_true, y_pred, max_val=1)
    return loss_function


class DCNN():
    def __init__(self):
        self.channels = 1
        self.lr_h = 64
        self.lr_w = 64
        self.lr_shape = (self.lr_h, self.lr_w, self.channels)

        self.dataset_name = '20191207_0450'

        adam = Adam(0.00003)

        self.generator = self.Generator()
        plot_model(self.generator, to_file='DCNN_model.png')

        self.generator.compile(optimizer=adam, loss='mse')

    def Generator(self):
        # 生成器
        def residual(input_layer, filters):
            r = Conv2D(filters, kernel_size=3, strides=1, padding='same')(input_layer)
            r = Activation('relu')(r)
            r = Conv2D(filters, kernel_size=3, strides=1, padding='same')(r)
            r = Add()([r, input_layer])
            return r

        img_lr = Input(shape=self.lr_shape)
        g1 = Conv2D(256, kernel_size=9, strides=1, padding='valid')(img_lr)
        g1 = Activation('relu')(g1)

        g2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(g1)
        g2 = residual(g2, 64)
        for _ in range(15):
            g2 = residual(g2, 64)

        gen_hr = Conv2D(self.channels, kernel_size=5, strides=1, padding='valid', activation='tanh')(g2)
        return Model(img_lr, gen_hr)

    class LossHistory(Callback):
        def __init__(self):
            super().__init__()
            self.val_loss = {'batch': [], 'epoch': []}
            self.losses = {'batch': [], 'epoch': []}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.val_loss['batch'].append(logs.get('val_loss'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.val_loss['epoch'].append(logs.get('val_loss'))

        def loss_plot(self, loss_type, _b, _dataset_name):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            plt.savefig('./Model/' + _dataset_name + '/loss_band%d.png' % _b)

    def train(self, epochs, batch_size=16, band=0):
        self.generator.summary()

        # 监视测试集的损失值 / 输出epoch模型保存信息 / 保存最佳模型 / 保存整个模型信息
        model_name = './Model/' + self.dataset_name + '/DCNN_' + str(band) + '.h5'
        checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min')
        history = self.LossHistory()
        callbacks_list = [checkpoint, history]

        # 训练生成器
        imgs_hr = np.load('./Datasets/' + self.dataset_name + '/train/imgs_hr.npy')[:, :, :, band]
        imgs_lr = np.load('./Datasets/' + self.dataset_name + '/train/imgs_lr.npy')[:, :, :, band]
        val_data = np.load('./Datasets/' + self.dataset_name + '/train/val_data.npy')[:, :, :, band]
        val_label = np.load('./Datasets/' + self.dataset_name + '/train/val_label.npy')[:, :, :, band]
        imgs_hr = np.expand_dims(imgs_hr, axis=3)
        imgs_lr = np.expand_dims(imgs_lr, axis=3)
        val_data = np.expand_dims(val_data, axis=3)
        val_label = np.expand_dims(val_label, axis=3)
        self.generator.fit(imgs_lr, imgs_hr, batch_size=batch_size, validation_data=(val_data, val_label),
                           callbacks=callbacks_list, shuffle=True, epochs=epochs, verbose=1)

        # 绘制loss曲线
        history.loss_plot('epoch', band, self.dataset_name)


if __name__ == '__main__':
    start_time = time.time()
    for b in range(3):
        gan = DCNN()
        gan.train(epochs=40, batch_size=36, band=b)
    end_time = time.time()
    print('Execution Time: ', end_time - start_time)
    # plot_model(vgg, to_file='model.png')
