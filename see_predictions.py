import tensorflow as tf
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

def weighted_pixelwise_crossentropy(y_true, y_pred):
    _EPSILON = 10e-8
    epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)

    # Clips tensor values to a specified min and max.
    # t: A Tensor.
    # clip_value_min: A 0 - D(scalar) Tensor, or a Tensor with the same shape as t.The minimum value to clip by.
    # clip_value_max: A 0 - D(scalar) Tensor, or a Tensor with the same shape as t.The maximum value to clip by

    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # tf.reduce_sum Computes the sum of elements across dimensions of a tensor.
    # tf.multiply(x, y) Returns x * y element-wise.
    # y_true.reshape
    # y_pred.reshape
    beta = 0.6
    ones = tf.ones_like(y_true)
    pos_ce = tf.multiply(y_true * tf.log(y_pred), beta)
    neg_ce = tf.multiply((ones - y_true) * tf.log(ones - y_pred), 1-beta)
    pixelwise_bce = tf.reduce_sum(pos_ce + neg_ce)
    return (-1)*pixelwise_bce


def predictions():
    img_path = '/home/elsa/Desktop/Image_Saliency/msra/test/images/images/'
    mask_path = '/home/elsa/Desktop/Image_Saliency/msra/test/masks/masks/'

    rescale = 1./255
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    mask_files = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

    img_files.sort()
    mask_files.sort()

    # load model from json file
    json_file = open('vgg16_saliency_rescaled.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("low_res_weights/vgg16_saliency_epochs_7.h5")
    print("Loaded model from disk")
    # loaded_model.get_layer('input_1')(shape=None)
    # compile model
    loaded_model.compile(loss=weighted_pixelwise_crossentropy, optimizer='rmsprop')

    # predict on 10 random images
    for n in range(10):
        index = np.random.randint(low=0, high=len(img_files))
        print(index)
        print(img_files[index])
        input_img = Image.open(img_path + img_files[index])
        input_img = input_img.resize((128, 128), Image.BILINEAR)

        input_img = np.asarray(input_img, dtype=np.uint8)
        mask_size = (32, 32)
        plt.imshow(input_img)
        plt.show()

        mask = Image.open(mask_path + mask_files[index])
        mask = mask.resize(mask_size, Image.BILINEAR)
        mask = np.asarray(mask, dtype=np.bool)

        # show ground truth
        plt.imshow(mask)
        plt.show()

        # show prediction
        w, h, c = input_img.shape
        input_img = input_img.reshape(1, w, h, c)
        pred_raw = loaded_model.predict(input_img) * 255
        pred = pred_raw[0,:, :, 0].astype('uint8') # pred has shape (1, 32, 32, 1) to include batch size and number of channels
        plt.imshow(pred)
        plt.show()

        # false positive rate = % of dark pixels falsely labelled as 1
        # detect fpr mask using mask - pred (negative numbers correspond to fpr, positive numbers correspond to fnr
        fpr_mask = np.clip(pred_raw - mask, 0, 1) # clip the negative values
        fpr = np.sum(fpr_mask)
        print('fpr ', fpr)

        fnr_mask = np.clip(mask - pred_raw, 0, 1)
        fnr = np.sum(fnr_mask)
        print('fnr', fnr)

def evaluate():
    img_path = '/home/elsa/Desktop/Image_Saliency/msra/test/images/images/'
    mask_path = '/home/elsa/Desktop/Image_Saliency/msra/test/masks/masks/'

    rescale = 1. / 255
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    mask_files = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

    img_files.sort()
    mask_files.sort()

    # load model from json file
    json_file = open('vgg16_saliency_rescaled.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("low_res_weights/vgg16_saliency_epochs_7.h5")
    print("Loaded model from disk")

    # compile model
    loaded_model.compile(loss=weighted_pixelwise_crossentropy, optimizer='rmsprop')

    fpr = 0
    fnr = 0
    # predict on 200 random images
    for n in range(20):
        index = np.random.randint(low=0, high=len(img_files))
        print(index)
        print(img_files[index])
        input = Image.open(img_path + img_files[index])
        input = input.resize((128, 128), Image.BILINEAR)

        input = np.asarray(input, dtype=np.uint8)
        # input = np.transpose(input, (1, 0, 2))
        # plt.imshow(input)
        # plt.show()

        mask = Image.open(mask_path + mask_files[index])
        mask = mask.resize((32, 32), Image.BILINEAR)
        mask = np.asarray(mask, dtype=np.bool)

        # show ground truth
        # plt.imshow(mask)
        # plt.show()

        # show prediction
        w, h, c = input.shape
        input = input.reshape(1, w, h, c)
        pred_raw = loaded_model.predict(input)[0, :, :, 0]
        pred = (pred_raw*255).astype('uint8')  # pred has shape (1, 32, 32, 1) to include batch size and number of channels
        # plt.imshow(pred)
        # plt.show()

        # false positive rate = % of dark pixels falsely labelled as 1
        # detect fpr mask using mask - pred (negative numbers correspond to fpr, positive numbers correspond to fnr
        fpr_mask = np.clip(pred_raw - mask, 0, 1)  # clip the negative values
        fpr += np.sum(fpr_mask)/20

        fnr_mask = np.clip(mask - pred_raw, 0, 1)
        fnr += np.sum(fnr_mask)/20

    print('fpr = ', fpr)
    print('fnr = ', fnr)

if __name__=="__main__":
    predictions()
    evaluate()