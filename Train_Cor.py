"""
功能：
    训练+生成模型

v1.0 王剑 2020/02/27
"""

from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import h5py


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type, b):
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
        plt.savefig('./Train/Model/loss_band%d.png' % b)


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


def model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()        # 线性模型
    # nb_row为滤波器尺寸，可以调整
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(64, 64, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)      # 学习率
    # compile配置模型的学习过程，参数有优化器，损失函数，评估模型在训练和测试时的网络性能的指标
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train(_b):
    srcnn_model = model()
    print(srcnn_model.summary())
    train_name = './Train/Model/Train' + str(_b) + '.h5'
    CV_name = './Train/Model/CV' + str(_b) + '.h5'
    data, label = read_training_data(train_name)
    val_data, val_label = read_training_data(CV_name)

    # 监视测试集的损失值 / 输出epoch模型保存信息 / 保存最佳模型 / 保存整个模型信息
    model_name = './Train/Model/SRCNN' + str(_b) + '.h5'
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    history = LossHistory()
    callbacks_list = [checkpoint, history]

    # validation_data为True时在每次epoch后检测是否过拟合等问题
    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=1)

    # 绘制loss曲线
    history.loss_plot('epoch', _b)


if __name__ == "__main__":
    import time
    start = time.time()
    for band in range(3):
        train(band)
    end = time.time()
    print('Execution Time: ', end - start)
