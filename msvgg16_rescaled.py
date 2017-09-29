from keras.applications import vgg16
from keras.layers.convolutional import Conv2D, AtrousConv2D
from keras.layers.merge import Concatenate, Multiply, Add, Dot
from keras.layers.noise import AlphaDropout
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join


class ValCallback(Callback):
    def __init__(self, test_data_gen):
        self.test_data_gen = test_data_gen

    def on_epoch_end(self, epoch, logs={}):
        loss = self.model.evaluate_generator(self.test_data_gen,
                                             steps=1,
                                             use_multiprocessing=True)

        print('\nTesting loss: {} \n'.format(loss))

        # see a few predictions
        img_path = '/home/elsa/Desktop/Image_Saliency/msra/test/images/images/'
        mask_path = '/home/elsa/Desktop/Image_Saliency/msra/test/masks/masks/'

        rescale = 1. / 255
        img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        mask_files = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

        img_files.sort()
        mask_files.sort()

        for n in range(3):
            index = np.random.randint(low=0, high=len(img_files))
            print(index)
            print(img_files[index])
            input = Image.open(img_path + img_files[index])
            input = input.resize((128, 128), Image.BILINEAR)

            input = np.asarray(input, dtype=np.uint8)
            plt.imshow(input)
            plt.show()

            mask = Image.open(mask_path + mask_files[index])
            mask = mask.resize((32, 32), Image.BILINEAR)
            mask = np.asarray(mask, dtype=np.bool)

            # show ground truth
            plt.imshow(mask)
            plt.show()

            # show prediction
            input = input / 255
            w, h, c = input.shape
            input = input.reshape(1, w, h, c)
            pred = self.model.predict(input) * 255
            # loss = weighted_pixelwise_crossentropy(mask, pred) * 64
            pred = pred[0, :, :, 0].astype('uint8')
            # print('loss = ', loss)
            plt.imshow(pred)
            plt.show()


def weighted_pixelwise_crossentropy(y_true, y_pred):
    _EPSILON = 10e-8
    batch_size = 64
    epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)

    # Clips tensor values to a specified min and max.
    # t: A Tensor.
    # clip_value_min: A 0 - D(scalar) Tensor, or a Tensor with the same shape as t.The minimum value to clip by.
    # clip_value_max: A 0 - D(scalar) Tensor, or a Tensor with the same shape as t.The maximum value to clip by

    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # tf.reduce_sum Computes the sum of elements across dimensions of a tensor.
    # tf.multiply(x, y) Returns x * y element-wise.

    beta = 0.55
    ones = tf.ones_like(y_true)
    pos_ce = tf.multiply(y_true * tf.log(y_pred), beta)
    neg_ce = tf.multiply((ones - y_true) * tf.log(ones - y_pred), 1 - beta)
    pixelwise_bce = tf.reduce_sum(pos_ce + neg_ce) / batch_size
    return (-1) * pixelwise_bce

def minus_mean_rgb(input):
    mean_pixel = [103.939, 116.779, 123.68]
    mean_rgb_mask = np.ones(input.shape)
    mean_rgb_mask[:, :, 0] *= mean_pixel[0]
    mean_rgb_mask[:, :, 1] *= mean_pixel[1]
    mean_rgb_mask[:, :, 2] *= mean_pixel[2]
    # plt.imshow(input-mean_rgb_mask)
    # plt.show()
    return input - mean_rgb_mask

def ms_fcnn():
    mean_rgb = [103.939, 116.779, 123.68]

    # we create two instances with the same arguments
    image_gen_args = dict(featurewise_center=False,
                          featurewise_std_normalization=False,
                          rotation_range=90.0,
                          width_shift_range=0,
                          height_shift_range=0,
                          zoom_range=0,
                          preprocessing_function=minus_mean_rgb)

    mask_gen_args = dict(samplewise_center=False,
                         samplewise_std_normalization=False,
                         rotation_range=90.0,
                         width_shift_range=0,
                         height_shift_range=0,
                         zoom_range=0,
                         rescale=1./255)

    image_datagen = ImageDataGenerator(**image_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    seed = 1
    image_generator = image_datagen.flow_from_directory(
        # '/home/riachichristina/msra/train/images/',
        '/home/elsa/Desktop/Image_Saliency/msra/train/images/',
        class_mode=None,
        target_size=(128, 128),
        batch_size=64,
        seed=seed,
        shuffle=False)

    mask_generator = mask_datagen.flow_from_directory(
        # '/home/riachichristina/msra/train/resized_masks/',
        '/home/elsa/Desktop/Image_Saliency/msra/train/masks/',
        class_mode=None,
        color_mode='grayscale',
        target_size=(32, 32),
        batch_size=64,
        seed=seed,
        shuffle=False)

    print('zipping images')
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    val_image_generator = image_datagen.flow_from_directory(
        '/home/elsa/Desktop/Image_Saliency/msra/test/images/',
        class_mode=None,
        target_size=(128, 128),
        batch_size = 64,
        seed=seed,
        shuffle=False)

    val_mask_generator = mask_datagen.flow_from_directory(
        '/home/elsa/Desktop/Image_Saliency/msra/test/masks/',
        class_mode=None,
        color_mode='grayscale',
        target_size=(32, 32),
        batch_size=64,
        seed=seed,
        shuffle=False)

    val_generator = zip(val_image_generator, val_mask_generator)

    print('building model')
    base_model = vgg16.VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(128, 128, 3))

    # construct the saliency detection layers
    conv_scale_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(
        base_model.get_layer('block1_pool').output)
    conv_scale_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_1)
    conv_scale_1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_1)

    # zero padding
    conv_scale_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        base_model.get_layer('block2_pool').output)
    conv_scale_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_2)
    conv_scale_2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_2)

    # for the last three layers, skip subsampling and use atrous convolution
    conv_scale_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        base_model.get_layer('block3_conv3').output)
    conv_scale_3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_3)
    conv_scale_3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_3)

    # rewrite block 4 of vgg16 with atrous convolution
    # first get the weights of the 3 conv layers:
    block4_conv1_weights = base_model.get_layer('block4_conv1').get_weights()
    block4_conv2_weights = base_model.get_layer('block4_conv2').get_weights()
    block4_conv3_weights = base_model.get_layer('block4_conv3').get_weights()

    vgg16_block4_atrous1 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(2, 2),
                                  padding='same', trainable=False)(base_model.get_layer('block3_conv3').output)

    vgg16_block4_atrous2 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(2, 2),
                                  padding='same', trainable=False)(vgg16_block4_atrous1)

    vgg16_block4_atrous3 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(2, 2),
                                  padding='same', trainable=False)(vgg16_block4_atrous2)

    conv_scale_4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')\
        (vgg16_block4_atrous3)
    conv_scale_4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_4)
    conv_scale_4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_scale_4)

    # rewrite block_5 of vgg16 with atrous convolution
    # first get the weights of the 3 conv layers:
    block5_conv1_weights = base_model.get_layer('block5_conv1').get_weights()
    block5_conv2_weights = base_model.get_layer('block5_conv2').get_weights()
    block5_conv3_weights = base_model.get_layer('block5_conv3').get_weights()

    # SKIPPED POOLING LAYER
    vgg16_block5_atrous1 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(4, 4),
                                  padding='same', trainable=False)(vgg16_block4_atrous3)

    vgg16_block5_atrous2 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(4, 4),
                                  padding='same', trainable=False)(vgg16_block5_atrous1)

    vgg16_block5_atrous3 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', dilation_rate=(4, 4),
                                  padding='same', trainable=False)(vgg16_block5_atrous2)

    # SKIPPED POOLING LAYER
    conv_scale_5 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(
        vgg16_block5_atrous3)
    conv_scale_5 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_5)
    conv_scale_5 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv_scale_5)
    conv_scale_5 = AlphaDropout(rate=0.5, noise_shape=None, seed=None)(conv_scale_5)

    # aggregate the outputs of the 5 convolutional layers
    concat = Concatenate(axis=-1)
    pre_final = concat([conv_scale_1, conv_scale_2, conv_scale_3, conv_scale_4, conv_scale_5])
    # final layer uses sigmoid activation
    final = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(pre_final)
    end2end = Model(inputs=base_model.input, outputs=final)

    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    end2end.compile(optimizer='adam', loss=weighted_pixelwise_crossentropy)
    plot_model(end2end, show_shapes=1, to_file='ms_fcnn.png')

    # set the weights for the layers with atrous convolution
    end2end.get_layer('conv2d_10').set_weights(block4_conv1_weights)
    end2end.get_layer('conv2d_11').set_weights(block4_conv2_weights)
    end2end.get_layer('conv2d_12').set_weights(block4_conv3_weights)

    end2end.get_layer('conv2d_16').set_weights(block5_conv1_weights)
    end2end.get_layer('conv2d_17').set_weights(block5_conv2_weights)
    end2end.get_layer('conv2d_18').set_weights(block5_conv3_weights)

    # serialize model to JSON
    model_json = end2end.to_json()
    with open("vgg16_saliency_rescaled.json", "w") as json_file:
        json_file.write(model_json)

    print('training model')
    end2end.fit_generator(
        train_generator,
        use_multiprocessing=True,
        steps_per_epoch=140,
        epochs=3,
        verbose=1,
        callbacks=[ValCallback((val_generator))])

    end2end.save_weights('vgg16_saliency_epochs_3_55.h5')


if __name__ == "__main__":
    ms_fcnn()