"""
converts mask labels to categorical
saves training data in a giant array in a npy file
"""
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

def gen_categorical_masks():

    subcat_names = ['background', 'person', 'car', 'airplane', 'bus', 'train', 'truck',
                    'bird', 'cat', 'dog']

    train_annotation_file = '/home/riachielsa/mscoco/annotations/instances_train2017.json'

    #train_annotation_file = '/home/elsa/Desktop/COCO/annotations/instances_train2017.json'
    train_coco = COCO(train_annotation_file)

    # get id's from category list
    coco_cat_ids = train_coco.getCatIds(catNms=subcat_names)
    if coco_cat_ids[0] != 0:
        coco_cat_ids.insert(0, 0)
    print(coco_cat_ids)

    train_masks_path = '/home/riachielsa/mscoco/train/masks/'
    val_masks_path = '/home/riachielsa/mscoco/val/masks/'
    train_imgs_path = '/home/riachielsa/mscoco/train/imgs/'
    val_imgs_path = '/home/riachielsa/mscoco/val/imgs/'

    mask_size = (256, 256)
    img_size = (256, 256)
    n_classes = len(subcat_names)
    print(n_classes)

    n_train = 45000

    train_masks = [f for f in listdir(train_masks_path) if isfile(join(train_masks_path, f))]
    val_masks = [f for f in listdir(val_masks_path) if isfile(join(val_masks_path, f))]

    train_imgs = [f for f in listdir(train_imgs_path) if isfile(join(train_imgs_path, f))]
    val_imgs = [f for f in listdir(val_imgs_path) if isfile(join(val_imgs_path, f))]

    print(len(train_masks), len(train_imgs))
    train_mask_arr = np.zeros((n_train, mask_size[0], mask_size[1], n_classes), dtype=np.uint8)
    train_imgs_arr = np.zeros((n_train, img_size[0], img_size[1], 3))

    # sort the images and masks to correspond them by index
    train_imgs.sort()
    val_imgs.sort()
    train_masks.sort()
    val_masks.sort()

    # mean rgb values on Imagenet as described by VGG16 people
    mean_pixel = [103.939, 116.779, 123.68]

    mean_img = np.ones((img_size[0], img_size[1], 3))
    mean_img[:, :, 0] = mean_pixel[0]
    mean_img[:, :, 1] = mean_pixel[1]
    mean_img[:, :, 2] = mean_pixel[2]

    for index in range(n_train):
        file = train_masks[index]
        filename = file.split('.')[0]
        mask = Image.open(train_masks_path+file)
        mask = mask.resize(mask_size, Image.NEAREST)
        mask = np.asarray(mask, dtype=np.uint8)
        mask = mask % 255
        one_hot = np.zeros((mask_size[0], mask_size[1], n_classes), dtype=np.uint8)
        # save the mask as a one hot encoded ground truth
        for new_cat_id, coco_cat_id in enumerate(coco_cat_ids):
            bin_mask = mask == coco_cat_id
            one_hot[bin_mask, new_cat_id] = 1

        # everywhere else is background
        ix = np.logical_not(np.isin(mask, coco_cat_ids))
        one_hot[ix, 0] = 1

        train_mask_arr[index] = one_hot

        img = Image.open(train_imgs_path+filename+".jpg")
        img = img.resize(img_size, Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.asarray(img) - mean_img
        train_imgs_arr[index] = img

    np.savez_compressed('train_masks_full_large', train_mask_arr)
    np.savez_compressed('train_imgs_large', train_imgs_arr)


    added_val = 5000
    n_val = len(val_imgs) + added_val
    print(n_val)

    val_mask_arr = np.zeros((n_val, mask_size[0], mask_size[1], n_classes), dtype=np.uint8)
    val_imgs_arr = np.zeros((n_val, img_size[0], img_size[1], 3))

    for index in range(len(val_masks)):
        file = val_masks[index]
        filename = file.split('.')[0]
        mask = Image.open(val_masks_path+file)
        mask = mask.resize(mask_size, Image.NEAREST)
        mask = np.asarray(mask, dtype=np.uint8)
        mask = mask % 255
        one_hot = np.zeros((mask_size[0], mask_size[1], n_classes), dtype=np.uint8)
        # save the mask as a one hot encoded ground truth
        for new_cat_id, coco_cat_id in enumerate(coco_cat_ids):
            bin_mask = mask == coco_cat_id
            one_hot[bin_mask, new_cat_id] = 1

        # everywhere else is background
        ix = np.logical_not(np.isin(mask, coco_cat_ids))
        one_hot[ix, 0] = 1

        val_mask_arr[index] = one_hot

        img = Image.open(val_imgs_path+filename+".jpg")
        img = img.resize(img_size, Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.asarray(img) - mean_img
        val_imgs_arr[index] = img

    # add more images to the val set from the remaining images in the training directory
    for i, index in enumerate(range(len(val_masks), n_val)):

        file = train_masks[n_train + i]
        filename = file.split('.')[0]
        mask = Image.open(train_masks_path + file)
        mask = mask.resize(mask_size, Image.NEAREST)
        mask = np.asarray(mask, dtype=np.uint8)
        mask = mask % 255
        one_hot = np.zeros((mask_size[0], mask_size[1], n_classes), dtype=np.uint8)
        # save the mask as a one hot encoded ground truth
        for new_cat_id, coco_cat_id in enumerate(coco_cat_ids):
            bin_mask = mask == coco_cat_id
            one_hot[bin_mask, new_cat_id] = 1

        # everywhere else is background
        ix = np.logical_not(np.isin(mask, coco_cat_ids))
        one_hot[ix, 0] = 1

        val_mask_arr[index] = one_hot

        img = Image.open(train_imgs_path + filename + ".jpg")
        img = img.resize(img_size, Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.asarray(img) - mean_img
        val_imgs_arr[index] = img

    np.savez_compressed('val_masks_full_large', val_mask_arr)
    np.savez_compressed('val_imgs_large', val_imgs_arr)


if __name__=="__main__":
    gen_categorical_masks()
    # test_npy_files()
