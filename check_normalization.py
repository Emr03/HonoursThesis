import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

train_img_path = '/home/elsa/Desktop/Image_Saliency/msra/train/images/images/'
test_img_path = '/home/elsa/Desktop/Image_Saliency/msra/test/images/images/'

norm_train_img_path = '/home/elsa/Desktop/Image_Saliency/msra/train/norm_images/images/'
norm_test_img_path = '/home/elsa/Desktop/Image_Saliency/msra/test/norm_images/images/'

img_files = [f for f in listdir(train_img_path) if isfile(join(train_img_path, f))]
mean_pixel = [103.939, 116.779, 123.68]

for f in img_files:
    input = Image.open(train_img_path + f)
    input = np.asarray(input, dtype=np.uint8)
    plt.imshow(input)
    plt.show()

    mean_rgb_mask = np.ones(input.shape)
    mean_rgb_mask[:, :, 0] *= mean_pixel[0]
    mean_rgb_mask[:, :, 1] *= mean_pixel[1]
    mean_rgb_mask[:, :, 2] *= mean_pixel[2]

    plt.imshow(mean_rgb_mask)
    plt.show()

    plt.imshow(input - mean_rgb_mask)
    plt.show()
