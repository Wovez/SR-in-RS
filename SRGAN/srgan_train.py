"""
训练SRGAN模型

王剑
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Input, Dense
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow import concat


class SRGAN():
    def __init__(self):
        self.channels = 1
        self.lr_h = 64
        self.lr_w = 64
        self.lr_shape = (self.lr_h, self.lr_w, self.channels)

        self.hr_h = self.lr_h * 4
        self.hr_w = self.lr_w * 4
        self.hr_shape = (self.hr_h, self.hr_w, self.channels)

        patch = int(self.hr_h / 2 ** 4)
        self.D_patch = (patch, patch, 1)

        self.dataset_name = '20191207_0450'

        self.losses = {'epoch': [], 'G_loss': [], 'feature_loss': [], 'mean_loss': []}

        adam = Adam(0.0002, 0.5)
        self.vgg = self.subvgg()
        self.vgg.trainable = False

        self.discriminator = self.Discriminator()
        self.discriminator.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        plot_model(self.discriminator, to_file='discriminator_model.png')

        self.generator = self.Generator()
        plot_model(self.generator, to_file='generator_model.png')

        img_lr = Input(shape=self.lr_shape)
        gen_hr = self.generator(img_lr)
        gen_hr_vgg = concat([gen_hr, gen_hr, gen_hr], axis=3)
        gen_features = self.vgg(gen_hr_vgg)

        self.discriminator.trainable = False
        dis = self.discriminator(gen_hr)

        self.combined = Model(img_lr, [dis, gen_features, gen_hr])
        self.combined.compile(optimizer=adam, loss=['binary_crossentropy', 'mse', 'mean_absolute_error'],
                              loss_weights=[5e-1, 1, 1])

    def Generator(self):
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

        img_lr = Input(shape=self.lr_shape)
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
        gen_hr = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(g3)
        return Model(img_lr, gen_hr)

    def Discriminator(self):
        # 判别器
        def sub_block(input_layer, filters, strides):
            s = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input_layer)
            s = LeakyReLU(alpha=0.2)(s)
            s = BatchNormalization(momentum=0.8)(s)
            return s

        img_input = Input(shape=self.hr_shape)
        d1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(img_input)
        d1 = LeakyReLU(alpha=0.2)(d1)

        d2 = sub_block(d1, 64, 2)
        d2 = sub_block(d2, 128, 1)
        d2 = sub_block(d2, 128, 2)
        d2 = sub_block(d2, 256, 1)
        d2 = sub_block(d2, 256, 2)
        d2 = sub_block(d2, 512, 1)
        d2 = sub_block(d2, 512, 2)

        d3 = Dense(1024)(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        dis = Dense(1, activation='sigmoid')(d3)
        return Model(img_input, dis)

    def subvgg(self):
        vgg = VGG19(input_shape=(self.hr_h, self.hr_w, 3), include_top=False, weights='imagenet')
        vgg.outputs = [vgg.layers[9].output]
        img_input = Input(shape=(self.hr_h, self.hr_w, 3))
        img_features = vgg(img_input)
        return Model(img_input, img_features)

    def load_data(self, batch_size=1, band=0):
        imgs_hr = np.load('./Datasets/' + self.dataset_name + '/train/imgs_hr.npy')
        imgs_lr = np.load('./Datasets/' + self.dataset_name + '/train/imgs_lr.npy')
        idx = np.random.choice(imgs_hr.shape[0], size=batch_size, replace=False)
        img_hr = []
        img_lr = []
        for i in range(batch_size):
            _hr = np.zeros((imgs_hr.shape[1], imgs_hr.shape[2], 1))
            _lr = np.zeros((imgs_lr.shape[1], imgs_lr.shape[2], 1))
            _hr[:, :, 0] = imgs_hr[idx[i], :, :, band]
            _lr[:, :, 0] = imgs_lr[idx[i], :, :, band]
            img_hr.append(_hr)
            img_lr.append(_lr)
        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)
        return img_hr, img_lr

    def scheduler(self, models):
        # 学习率下降
        lr = 0
        for model in models:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
        print('lr changed to {}'.format(lr * 0.5))

    def loss(self, band):
        # 保存loss曲线
        plt.figure()
        # loss
        plt.plot(self.losses['epoch'], self.losses['G_loss'], 'g', label='G loss')
        plt.plot(self.losses['epoch'], self.losses['feature_loss'], 'k', label='feature loss')
        plt.plot(self.losses['epoch'], self.losses['mean_loss'], 'r', label='mean loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.ylim(0, 40)
        plt.savefig('./Model/' + self.dataset_name + '/loss_band%d.png' % band)

    def train(self, epochs, init_epoch=0, batch_size=1, save_interval=500, band=0):
        start_time = time.time()
        self.generator.summary()
        self.discriminator.summary()
        g_loss = []
        temp = [float('inf'), float('inf'), float('inf')]
        if init_epoch != 0:
            self.generator.load_weights('./Model/%s/gen_epoch%d_%d.h5' % (self.dataset_name, init_epoch, band))
            self.discriminator.load_weights('./Model/%s/dis_epoch%d_%d.h5' % (self.dataset_name, init_epoch, band))
            self.losses = np.load('./Model/%s/loss_%d.npy' % (self.dataset_name, band), allow_pickle=True).item()
            temp = [self.losses['G_loss'], self.losses['feature_loss'], self.losses['mean_loss']]
            init_epoch += 1
        for epoch in range(init_epoch, epochs):
            if epoch % 20000 == 0 and epoch != 0:
                self.scheduler([self.combined, self.discriminator])
            # 训练判别器
            imgs_hr, imgs_lr = self.load_data(batch_size=batch_size, band=band)
            fake_hr = self.generator.predict(imgs_lr)
            valid = np.ones((batch_size,) + self.D_patch)
            fake = np.zeros((batch_size,) + self.D_patch)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            imgs_hr, imgs_lr = self.load_data(batch_size=batch_size, band=band)
            imgs_hr = np.concatenate([imgs_hr, imgs_hr, imgs_hr], axis=3)
            valid = np.ones((batch_size,) + self.D_patch)
            image_features = self.vgg.predict(imgs_hr)
            g_loss = self.combined.train_on_batch(imgs_lr, [valid, image_features, imgs_hr])
            print(d_loss, g_loss)
            elapsed_time = time.time() - start_time
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f, mean loss: %05f] time: %s "
                  % (epoch, epochs, d_loss[0], 100 * d_loss[1], g_loss[1], g_loss[2], g_loss[3], elapsed_time))

            # 保存loss值
            if epoch % 10 == 0:
                self.losses['epoch'].append(epoch)
                self.losses['G_loss'].append(g_loss[1])
                self.losses['feature_loss'].append(g_loss[2])
                self.losses['mean_loss'].append(g_loss[3])
            
            # 保存最佳模型
            if (temp[0] > g_loss[1]) and (temp[1] > g_loss[2]) and (temp[2] > g_loss[3]):
                self.generator.save_weights('./Model/%s/gen_best_%d.h5' % (self.dataset_name, band))
                self.discriminator.save_weights('./Model/%s/dis_best_%d.h5' % (self.dataset_name, band))
                temp = [g_loss[1], g_loss[2], g_loss[3]]
                print('Save current best model (epoch = %d)' % epoch)

            # 保存模型
            if (epoch % save_interval == 0) & (epoch != 0):
                self.generator.save_weights('./Model/%s/gen_epoch%d_%d.h5' % (self.dataset_name, epoch, band))
                self.discriminator.save_weights('./Model/%s/dis_epoch%d_%d.h5' % (self.dataset_name, epoch, band))
                print('Save current model (epoch = %d)' % epoch)
        self.losses['epoch'].append(epochs)
        self.losses['G_loss'].append(g_loss[1])
        self.losses['feature_loss'].append(g_loss[2])
        self.losses['mean_loss'].append(g_loss[3])
        self.loss(band=band)
        self.generator.save_weights('./Model/%s/gen_epoch%d_%d.h5' % (self.dataset_name, epochs, band))
        self.discriminator.save_weights('./Model/%s/dis_epoch%d_%d.h5' % (self.dataset_name, epochs, band))
        np.save('./Model/%s/loss_%d' % (self.dataset_name, band), self.losses)


if __name__ == '__main__':
    for b in range(3):
        gan = SRGAN()
        gan.train(epochs=10000, init_epoch=0, batch_size=2, save_interval=500, band=b)
    # plot_model(vgg, to_file='model.png')
