"""
Implements image segmentation
"""

from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.models import Model, model_from_json
from keras.layers import AlphaDropout, ZeroPadding2D, UpSampling2D, BatchNormalization
from keras.applications import VGG16
from keras.initializers import Orthogonal, VarianceScaling, he_uniform, zeros
import tensorflow as tf
import numpy as np
from keras.regularizers import l2
from keras.optimizers import Adam, Adadelta, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from keras.utils import multi_gpu_model, plot_model
from enum import Enum
from time import time
import statistics
import keras.backend as K


class MSCNN:
    def __init__(self):

        self.end2end = None

        # for training and validation
        self.train_imgs_file = None
        self.train_masks_file = None
        self.val_imgs_file = None
        self.val_masks_file = None
        self.train_binary_masks_file = None
        self.val_binary_masks_file = None

        self.batch_size = 32
        self.n_gpu = 2

        subcat_names = ['background', 'person', 'car',
                        'bird', 'cat', 'dog']

        self.class_weights = []

        self.n_classes = len(subcat_names)

        self.mask_size = 256

        self.initializer = he_uniform()
        self.bias_init = zeros()

        self.optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


    def categorical_crossentropy(self, y_true, y_pred):
        """
        computes the categorical crossentropy over the ground truth mask divided by size of mask
        :param y_true: 4D tensor of shape batch_size, w, h, n_classes
        :param y_pred:
        :return:
        """
        _EPSILON = 10e-8
        epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        L_mask = tf.multiply(y_true, tf.log(y_pred)) # returns shape batch_size, w, h, n_classes
        L_mask = tf.reduce_sum(L_mask, axis=[1, 2])

        L_mask = L_mask * self.class_weights[tf.newaxis, :]

        return -tf.reduce_sum(L_mask) / (self.mask_size**2 * self.batch_size)

    def load_train_data(self, train_imgs_file, train_masks_file, compute_class_weights):

        self.train_imgs_file = train_imgs_file
        self.train_masks_file = train_masks_file

        print('loading training data')
        self.train_imgs = np.load(self.train_imgs_file)['arr_0']
        self.train_masks = np.load(self.train_masks_file)['arr_0']

        self.train_size = self.train_imgs.shape[0]
        print('training data is loaded', self.train_size, self.train_masks.shape[0])

        pixel_freq = [0.0 for i in range(self.n_classes)]
        img_freq = [0 for i in range(self.n_classes)]

        if compute_class_weights==True:

            for m in range(self.train_size):
                for c in range(self.n_classes):
                    c_freq = np.count_nonzero(self.train_masks[m, :, :, c])
                    if c_freq > 0:
                        img_freq[c] += 1
                        pixel_freq[c] += c_freq

            for c in range(self.n_classes):
                pixel_freq[c] /= (img_freq[c] * self.mask_size**2)

            median = statistics.median(pixel_freq)

            self.class_weights = []
            for c in range(self.n_classes):
                self.class_weights.append(median / pixel_freq[c])

            np.save('class_weights.npy', np.array(self.class_weights))
            self.class_weights = tf.constant(self.class_weights)

        else:
            self.class_weights = np.load('class_weights.npy')

    def load_val_data(self, val_imgs_file, val_masks_file):

        self.val_imgs_file = val_imgs_file
        self.val_masks_file = val_masks_file

        print('loading validation data')
        self.val_imgs = np.load(self.val_imgs_file)['arr_0']
        self.val_masks = np.load(self.val_masks_file)['arr_0'].astype(np.uint8)

        self.val_size = self.val_imgs.shape[0]
        print('validation data is loaded')

    def build_segmentation_model(self):

        print('building model')

        base_model = VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(256, 256, 3))

        conv_scale_5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(base_model.get_layer('block5_conv3').output)

        conv_scale_5 = Conv2D(name='conv_scale_5_1', filters=128, kernel_size=(3, 3), activation=None, padding='same',
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init)(conv_scale_5)

        conv_scale_5 = LeakyReLU(alpha=0.3)(conv_scale_5)

        conv_scale_5 = Conv2D(name='conv_scale_5_2', filters=128, kernel_size=(3, 3), activation=None, padding='same',
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init)(conv_scale_5)

        conv_scale_5 = LeakyReLU(alpha=0.3)(conv_scale_5)

        conv_scale_5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv_scale_5)

        conv_scale_5 = UpSampling2D(size=(2, 2))(conv_scale_5)

        #################################################################################################################################################

        conv_scale_4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(base_model.get_layer('block4_conv3').output)

        conv_scale_4 = concatenate([conv_scale_4, conv_scale_5])

        conv_scale_4 = Conv2D(name='conv_scale_4_1', filters=128, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              padding='same', kernel_initializer=self.initializer, bias_initializer=self.bias_init)(conv_scale_4)

        conv_scale_4 = LeakyReLU(alpha=0.3)(conv_scale_4)

        conv_scale_4 = Conv2D(name='conv_scale_4_2', filters=128, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              padding='same', kernel_initializer=self.initializer)(conv_scale_4)

        conv_scale_4 = LeakyReLU(alpha=0.3)(conv_scale_4)

        conv_scale_4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv_scale_4)

        # upsample scale 4 to get feature map of size 64 x 64
        conv_scale_4 = UpSampling2D(size=(2, 2))(conv_scale_4)

        #################################################################################################################################################

        conv_scale_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)((base_model.get_layer('block3_conv3').output))

        conv_scale_3 = concatenate([conv_scale_3, conv_scale_4])

        conv_scale_3 = Conv2D(name='conv_scale_3_1', filters=64, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init, padding='same')(conv_scale_3)

        conv_scale_3 = LeakyReLU(alpha=0.3)(conv_scale_3)

        conv_scale_3 = Conv2D(name='conv_scale_3_2', filters=64, kernel_size=(3, 3), strides=(1, 1),activation=None,
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init, padding='same')(conv_scale_3)

        conv_scale_3 = LeakyReLU(alpha=0.3)(conv_scale_3)

        conv_scale_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv_scale_3)

        # upsample scale 3
        conv_scale_3 = UpSampling2D(size=(2, 2))(conv_scale_3)

        #################################################################################################################################################

        conv_scale_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)((base_model.get_layer('block2_conv2').output))

        conv_scale_2 = concatenate([conv_scale_2, conv_scale_3])

        conv_scale_2 = Conv2D(name='conv_scale_2_1', filters=64, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init, padding='same')(conv_scale_2)

        conv_scale_2 = LeakyReLU(alpha=0.3)(conv_scale_2)

        conv_scale_2 = Conv2D(name='conv_scale_2_2', filters=64, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init, padding='same')(conv_scale_2)

        conv_scale_2 = LeakyReLU(alpha=0.3)(conv_scale_2)

        conv_scale_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv_scale_2)

        # upsample scale 2
        conv_scale_2 = UpSampling2D(size=(2, 2))(conv_scale_2)

        #################################################################################################################################################

        conv_scale_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)((base_model.get_layer('block1_conv2').output))

        conv_scale_1 = concatenate([conv_scale_1, conv_scale_2])

        conv_scale_1 = Conv2D(name='conv_scale_1_1', filters=32, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              kernel_initializer=self.initializer, bias_initializer=self.bias_init,  padding='same')(conv_scale_1)

        conv_scale_1 = LeakyReLU(alpha=0.3)(conv_scale_1)

        conv_scale_1 = Conv2D(name='conv_scale_1_2', filters=32, kernel_size=(3, 3), strides=(1, 1), activation=None,
                              padding='same', kernel_initializer=self.initializer, bias_initializer=self.bias_init)(conv_scale_1)

        conv_scale_1 = LeakyReLU(alpha=0.3)(conv_scale_1)

        # final layer uses softmax activation
        final = Conv2D(name='softmax', filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1), activation='softmax',
                       kernel_initializer=self.initializer, bias_initializer=self.bias_init)(conv_scale_1)
        end2end = Model(inputs=base_model.input, outputs=final)

        # set VGG16 layers to not trainable
        for layer in base_model.layers:
            layer.trainable = False

        self.end2end = end2end
        # plot_model(end2end, show_shapes=1, to_file='recombinator_cnn.png')

        # serialize model to JSON
        model_json = self.end2end.to_json()
        with open("recombinator_model.json", "w") as json_file:
            json_file.write(model_json)

        self.parallel_model = multi_gpu_model(self.end2end, self.n_gpu)
        self.parallel_model.compile(optimizer=self.optimizer, loss=self.categorical_crossentropy)


    def load_from_json(self, json_filename, weights_filename):
        """
        :param json_filename: load model saved in json file
        :return: model
        """
        print('loading model from json file ', json_filename)
        json_file = open(json_filename, 'r')
        model_desc = json_file.read()
        json_file.close()
        model = model_from_json(model_desc)
        model.load_weights(weights_filename)

        return model

    def train(self):

        filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        csv_logger = CSVLogger('csv_log.csv', separator=',', append=False)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
        tensorboard = TensorBoard(log_dir="ini_logs/{}".format(time()), histogram_freq=10, batch_size=self.batch_size, write_grads=True, write_images=True, embeddings_freq=0)
        callbacks_list = [csv_logger, tensorboard, checkpoint]

        print('training model')

        self.parallel_model.fit(
            self.train_imgs,
            self.train_masks,
            batch_size=self.n_gpu * self.batch_size,
            validation_data=[self.val_imgs, self.val_masks],
            epochs=30,
            verbose=1,
            callbacks=callbacks_list
        )

        self.end2end.save_weights('new_weights.h5')

    def continue_training(self, weights_file):

        self.end2end = self.load_from_json('recombinator_model.json', weights_file)
        self.parallel_model = multi_gpu_model(self.end2end, self.n_gpu)
        self.parallel_model.compile(optimizer=self.optimizer, loss=self.categorical_crossentropy)
        self.train()


if __name__ == "__main__":
    model = MSCNN()
    model.load_train_data('train_imgs.npz', 'train_masks_full.npz', True)
    model.load_val_data('val_imgs.npz', 'val_masks_full.npz')
    model.build_segmentation_model()

    model.train()
