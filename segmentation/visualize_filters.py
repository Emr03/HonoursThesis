from recombinator_cnn import MSCNN
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns

K.set_learning_phase(0)

n_classes = 6

test_images = np.load('test_set_imgs.npy')
test_masks = np.load('test_set_masks.npy')

cnn = MSCNN()
model = cnn.load_from_json('recombinator_model.json', 'new_weights.h5')

get_scale_5 = K.function([model.get_layer('input_1').input], [model.get_layer('up_sampling2d_1').output])
get_scale_4 = K.function([model.get_layer('input_1').input], [model.get_layer('up_sampling2d_2').output])
get_scale_3 = K.function([model.get_layer('input_1').input], [model.get_layer('up_sampling2d_3').output])
get_scale_2 = K.function([model.get_layer('input_1').input], [model.get_layer('up_sampling2d_4').output])
get_scale_1 = K.function([model.get_layer('input_1').input], [model.get_layer('block1_conv2').output])

cmap = sns.palplot(sns.color_palette("hls", n_classes))

# mean rgb values on Imagenet as described by VGG16 people
mean_pixel = [103.939, 116.779, 123.68]

mean_img = np.ones((256, 256, 3))
mean_img[:, :, 0] = mean_pixel[0]
mean_img[:, :, 1] = mean_pixel[1]
mean_img[:, :, 2] = mean_pixel[2]

for i in range(test_images.shape[0]):

    input = test_images[i].reshape(1, 256, 256, 3)
    img = (input[0] + mean_img).astype(np.uint8, copy=False)

    ground_truth = np.argmax(test_masks[i], axis=-1)

    scale_5 = get_scale_5([input])

    #scale_5 = tf.reduce_sum(scale_5[0], axis=[0, -1])

    scale_4 = get_scale_4([input])

    scale_3 = get_scale_3([input])

    scale_2 = get_scale_2([input])

    scale_1 = get_scale_1([input])

    plt.imshow(img)
    plt.show()

    f, ax = plt.subplots(5, 8)
    for k in range(8):
        for l in range(5):
            ax[l, k].axis('Off')

    for j in range(8):
        ax[0, j].imshow(scale_5[0][0, :, :, j])


    for j in range(8):
        ax[1, j].imshow(scale_4[0][0, :, :, j])


    for j in range(8):
        ax[2, j].imshow(scale_3[0][0, :, :, j])


    for j in range(8):
        ax[3, j].imshow(scale_2[0][0, :, :, j])


    for j in range(8):
        ax[4, j].imshow(scale_1[0][0, :, :, j])

    plt.show()
