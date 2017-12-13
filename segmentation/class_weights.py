import numpy as np
import statistics
from os import listdir
from os.path import isfile, join
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

train_masks = np.load('train_masks_full.npz')['arr_0']
val_masks = np.load('val_masks_full.npz')['arr_0']

class_names = ['background', 'person', 'car',
                    'bird', 'cat', 'dog']

train_pixel_freq = [0.0 for i in range(6)]
train_img_freq = [0.0 for i in range(6)]

for m in range(train_masks.shape[0]):
    for c in range(6):
        c_freq = np.count_nonzero(train_masks[m, :, :, c])
        if c_freq > 0:
            train_img_freq[c] += 1
            train_pixel_freq[c] += c_freq / (256)**2

val_pixel_freq = [0.0 for i in range(6)]
val_img_freq = [0.0 for i in range(6)]

for m in range(val_masks.shape[0]):
    for c in range(6):
        c_freq = np.count_nonzero(val_masks[m, :, :, c])
        if c_freq > 0:
            val_img_freq[c] += 1
            val_pixel_freq[c] += c_freq / (256)**2

pixel_freq = pd.DataFrame(data = [train_pixel_freq, val_pixel_freq], columns=class_names,  index=['train', 'val'])
img_freq = pd.DataFrame(data = [train_img_freq, val_img_freq], columns=class_names, index=['train', 'val'])

pixel_freq.to_csv('pixel_freq.csv')
img_freq.to_csv('image_freq.csv')



# for c in range(6):
#     pixel_freq[c] /= (img_freq[c]*1024)

# median = statistics.median(pixel_freq)
#
# class_weights = []
# for c in range(6):
#     class_weights.append(median / pixel_freq[c])
#
# np.save('class_weights.npy', np.array(class_weights))
# print(class_weights)
