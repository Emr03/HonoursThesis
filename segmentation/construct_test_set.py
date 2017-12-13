
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from matplotlib import pyplot as plt

# load validation data
val_imgs = np.load("val_imgs.npz")['arr_0']
val_masks = np.load("val_masks_full.npz")['arr_0']

np.save('test_set_imgs', val_imgs[50:100])
np.save('test_set_masks', val_masks[50:100])

# load training data
# train_imgs = np.load('train_imgs.npz')['arr_0']
# train_masks = np.load('train_masks_full.npz')['arr_0']
#
# np.save('train_set_imgs', train_imgs[0:50])
# np.save('train_set_masks', train_masks[0:50])

