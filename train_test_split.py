from os import listdir
from os.path import isfile, join
import shutil

# img_path = '/home/riachichristina/msra/images/images/'
# mask_path = '/home/riachichristina/msra/masks/masks/'
#
# train_img_path = '/home/riachichristina/msra/train/images/images/'
# train_mask_path = '/home/riachichristina/msra/train/masks/masks'
#
# test_img_path = '/home/riachichristina/msra/test/images/images/'
# test_mask_path = '/home/riachichristina/msra/test/masks/masks/'

img_path = '/home/elsa/Desktop/Image_Segmentation/msra/
mask_path = '/home/elsa/Desktop/Image_Segmentation/msra/

train_img_path = '/home/elsa/Desktop/Image_Segmentation/msra/train/images/images'
train_mask_path = '/home/elsa/Desktop/Image_Segmentation/msra/train/masks/masks'

test_img_path = '/home/elsa/Desktop/Image_Segmentation/msra/test/images/images'
test_mask_path = '/home/elsa/Desktop/Image_Segmentation/msra/test/masks/masks'

img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
mask_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]

# do a 90%-10% split
train_set = []
test_set = []

counter = 0
for f in img_files:
    name = f.split('.')[0]
    counter+=1
    if counter <= 9000:
        shutil.move(img_path + f, train_img_path)
        shutil.move(mask_path + name + '.png', train_mask_path)
    else:
        shutil.move(img_path + f, test_img_path)
        shutil.move(mask_path + name + '.png', test_mask_path)


    