"""
Implements image segmentation on top of an image saliency network
"""

from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_json
from keras.layers import AlphaDropout
from keras.applications import VGG16
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback, LearningRateScheduler
from keras.utils import multi_gpu_model
from enum import Enum
from time import time
import statistics
import keras.backend as K


class ValCallback(Callback):
    def __init__(self, val_data, val_masks):
        self.val_data = val_data
        self.val_masks = val_masks

    def on_epoch_end(self, epoch, logs={}):
        loss = self.model.evaluate(self.val_data,
                                   self.val_masks,
                                   batch_size=64)

        print('\nValidation loss: {} \n'.format(loss))


class Mode(Enum):
    ATTENTION = "attention"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class MSCNN:
    def __init__(self, mode):

        self.attention_model = None
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

        if not isinstance(mode, Mode):
            raise ValueError("invalid type for mode")

        self.mode = mode
        if mode == Mode.ATTENTION:
            self.active_model = self.attention_model

        else:
            self.active_model = self.end2end

    def binary_crossentropy(self, y_true, y_pred):
        """
        class indifferent
        computes the binary pixelwise crossentropy
        :param y_true: 3D tensor of shape batch_size, w, h
        :param y_pred: 3D tensor of shape batch_size, w, h
        :return:
        """
        _EPSILON = 10e-8
        epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        beta = 0.5
        ones = tf.ones_like(y_true)
        pos_ce = tf.multiply(y_true * tf.log(y_pred), beta)
        neg_ce = tf.multiply((ones - y_true) * tf.log(ones - y_pred), 1 - beta)
        pixelwise_bce = tf.reduce_sum(pos_ce + neg_ce) / (32*32*self.batch_size)
        return (-1) * pixelwise_bce

    def categorical_crossentropy(self, y_true, y_pred):
        """
        mask indifferent
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

        return -tf.reduce_sum(L_mask) / (32*32*self.batch_size)

    def load_train_data(self, train_imgs_file, train_masks_file, compute_class_weights):

        self.train_imgs_file = train_imgs_file
        self.train_masks_file = train_masks_file

        print('loading training data')
        self.train_imgs = np.load(self.train_imgs_file)
        self.train_masks = np.load(self.train_masks_file)

        self.train_size = self.train_imgs.shape[0]
        print('training data is loaded')

        pixel_freq = [0 for i in range(21)]
        img_freq = [0 for i in range(21)]

        if self.mode == Mode.SEGMENTATION and compute_class_weights==True:

            for m in range(self.train_size):
                for c in range(self.n_classes):
                    c_freq = np.count_nonzero(self.train_masks[m, :, :, c])
                    if c_freq > 0:
                        img_freq[c] += 1
                        pixel_freq[c] += c_freq

            for c in range(self.n_classes):
                pixel_freq[c] /= (img_freq[c] * 1024)

            median = statistics.median(pixel_freq)

            self.class_weights = []
            for c in range(self.n_classes):
                self.class_weights.append(median / pixel_freq[c])

            self.class_weights = tf.constant(self.class_weights)
            np.save('class_weights.npy', np.array(self.class_weights))

        elif self.mode == Mode.SEGMENTATION:
            self.class_weights = np.load('class_weights.npy')

    def load_val_data(self, val_imgs_file, val_masks_file):

        self.val_imgs_file = val_imgs_file
        self.val_masks_file = val_masks_file

        print('loading validation data')
        self.val_imgs = np.load(self.val_imgs_file)[0:10000]
        self.val_masks = np.load(self.val_masks_file).astype(np.uint8)[0:10000]

        self.val_size = self.val_imgs.shape[0]
        print('validation data is loaded')

    def build_attention_model(self, json_file, weights_file):
        """
        build the saliency model from model saved in json file and load corresponding weights
        :param json_file:
        :param weights_file:
        :return:
        """

        print('building model')
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        attention_model = model_from_json(loaded_model_json)

        attention_model.load_weights(weights_file)
        print("Loaded model from disk")

        self.attention_model = attention_model
        self.attention_model.compile(optimizer='adam', loss=self.binary_crossentropy)
        self.active_model = self.attention_model

    def build_segmentation_model(self, attention_json_file):
        """
        builds a segmentation model based on a saliency model and pre-trained weights
        :param attention_json_file:
        :param attention_weights_file:
        :return:
        """

        num_classes = self.n_classes
        print('building model')

        base_model = VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(256, 256, 3))

        # construct the saliency detection layers
        # conv_scale_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(4, 4), activation='relu', padding='same')(
        #     base_model.get_layer('block1_pool').output)
        # conv_scale_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_1)
        # conv_scale_1 = Conv2D(filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_1)
        # conv_scale_1 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_1)
        #
        # # zero padding
        # conv_scale_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(
        #     base_model.get_layer('block2_pool').output)
        # conv_scale_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_2)
        # conv_scale_2 = Conv2D(filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_2)
        # conv_scale_5 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_2)

        conv_scale_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
            base_model.get_layer('block3_pool').output)
        conv_scale_3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_3)
        conv_scale_3 = Conv2D(filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_3)
        conv_scale_3 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_3)

        # for the last two layers, skip subsampling and use atrous convolution
        conv_scale_4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
            base_model.get_layer('block4_conv3').output)
        conv_scale_4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_4)
        conv_scale_4 = Conv2D(filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_4)
        conv_scale_4 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_4)

        # rewrite block_5 of vgg16 with atrous convolution
        # first get the weights of the 3 conv layers:
        block5_conv1_weights = base_model.get_layer('block5_conv1').get_weights()
        block5_conv2_weights = base_model.get_layer('block5_conv2').get_weights()
        block5_conv3_weights = base_model.get_layer('block5_conv3').get_weights()

        vgg16_block5_atrous1 = Conv2D(name='block5_1', filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(2, 2),
                                      padding='same', trainable=False)(base_model.get_layer('block4_conv3').output)

        vgg16_block5_atrous2 = Conv2D(name='block5_2', filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(2, 2),
                                      padding='same', trainable=False)(vgg16_block5_atrous1)

        vgg16_block5_atrous3 = Conv2D(name='block5_3', filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(2, 2),
                                      padding='same', trainable=False)(vgg16_block5_atrous2)

        conv_scale_5 = Conv2D(filters=128, kernel_size=(1, 1), dilation_rate=(4, 4), activation='relu', padding='same')(
            vgg16_block5_atrous3)
        conv_scale_5 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_5)
        conv_scale_5 = Conv2D(filters=self.n_classes, kernel_size=(1, 1), dilation_rate=(4, 4), activation='sigmoid')(conv_scale_5)
        conv_scale_5 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_5)

        # aggregate the outputs of the 5 convolutional layers
        concat = Concatenate(axis=-1)
        pre_final = concat([conv_scale_3, conv_scale_4, conv_scale_5])
        # final layer uses sigmoid activation
        final = Conv2D(filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(pre_final)
        end2end = Model(inputs=base_model.input, outputs=final)

        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        #self.optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0)

        # set the weights for the layers with atrous convolution
        end2end.get_layer('block5_1').set_weights(block5_conv1_weights)
        end2end.get_layer('block5_2').set_weights(block5_conv2_weights)
        end2end.get_layer('block5_3').set_weights(block5_conv3_weights)

        self.end2end = end2end
        self.parallel_model = multi_gpu_model(self.end2end, gpus=self.n_gpu)

        # serialize model to JSON
        model_json = self.end2end.to_json()
        with open("semantic_segmentation_model_2.json", "w") as json_file:
            json_file.write(model_json)

        self.optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
        self.parallel_model.compile(optimizer=self.optimizer, loss=self.categorical_crossentropy)
        self.active_model = self.end2end


    def load_from_json(self, json_filename, weights_filename):
        """
        :param json_filename: load model saved in json file
        :return:
        """
        print('loading model from json file ', json_filename)
        json_file = open(json_filename, 'r')
        model = json_file.read()
        json_file.close()
        self.end2end = model_from_json(model)
        self.end2end.load_weights(weights_filename)
        self.parallel_model = multi_gpu_model(self.end2end, gpus=self.n_gpu)

        if self.mode == Mode.ATTENTION:
            self.parallel_model.compile(optimizer='adam', loss=self.binary_crossentropy)

        else:
            self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            self.parallel_model.compile(optimizer=self.optimizer, loss=self.categorical_crossentropy)

        self.active_model = self.end2end
        print('model loaded from json file')

    def to_onehot(self, mask):
        one_hot = np.zeros((64, 64, self.n_classes), dtype=np.uint8)
        # save the mask as a one hot encoded ground truth
        for new_cat_id, coco_cat_id in enumerate(self.coco_cat_ids):
            bin_mask = mask == coco_cat_id
            one_hot[bin_mask, new_cat_id] = 1

        return one_hot

    def train(self):

        #filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        csv_logger = CSVLogger('csv_log_simpler.csv', separator=',', append=False)
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=50, batch_size=self.batch_size, write_grads=True,  embeddings_freq=0)
        callbacks_list = [csv_logger, tensorboard]

        print('training model')

        self.parallel_model.fit(
            self.train_imgs,
            self.train_masks,
            batch_size=self.n_gpu * self.batch_size,
            validation_data=[self.val_imgs, self.val_masks],
            epochs=300,
            verbose=1,
            callbacks=callbacks_list
        )

        self.end2end.save_weights('new_weights_simpler.h5')

    def continue_training(self, weights_file):

        self.load_from_json('semantic_segmentation_model_2.json', weights_file)
        self.train()


if __name__ == "__main__":
    model = MSCNN(Mode.CLASSIFICATION)
    #model.load_from_json('semantic_segmentation_model_1.json')

    model.load_train_data('train_imgs.npy', 'train_masks.npy', True)
    model.load_val_data('val_imgs.npy', 'val_masks.npy')
    model.build_segmentation_model('vgg16_saliency_final.json')
    model.train()

    #model.continue_training('new_weights_simpler.h5')
    