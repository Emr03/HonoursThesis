from __future__ import division, print_function, unicode_literals
import os
import errno
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from shutil import copyfile

def coco_json_to_segmentation(seg_mask_path, annFile, source_image_path, target_image_path, img_ids):
    """

    :param seg_mask_output_paths:
    :param annotation_paths:
    :param seg_mask_image_paths:
    :param verbose:
    :return:
    """

    print('Loading COCO Annotations File: ', annFile)
    print('Segmentation Mask Output Folder: ', seg_mask_path)
    print('Target Image Folder: ', target_image_path)

    coco = COCO(annFile)

    print('Converting Annotations to Segmentation Masks...')
    mkdir_p(seg_mask_path)
    total_imgs = len(img_ids)
    print(total_imgs)

    subcat_ids = coco.getCatIds(catNms=subcat_names)
    subcat_count = [0 for _ in range(len(subcat_ids))]

    for id in img_ids:
        img = coco.loadImgs([id])[0]
        h = img['height']
        w = img['width']
        name = img['file_name']
        root_name = name[:-4] # get the name without the extension
        filename = os.path.join(seg_mask_path, root_name + ".png")

        MASK = np.zeros((h, w), dtype=np.uint8)

        # loop over annotations (annotation is for each individual object in an image)
        # imgToAnn return a list of all the annotations for a given image id
        for ann in coco.imgToAnns[id]:

            if ann['category_id'] in subcat_ids:
                cat_idx = subcat_ids.index(ann['category_id'])
                subcat_count[cat_idx] += 1

            mask = coco.annToMask(ann)
            # get a binary array for where the mask > 0
            idxs = np.where(mask > 0)

            # set the non-zero values to the category id
            MASK[idxs] = ann['category_id']

        # aggregate all the individual masks in one image
        im = Image.fromarray(MASK)

        # save the image mask
        im.save(filename)

        # save the image in the directory for training images
        img_name = root_name+'.jpg'
        copyfile(os.path.join(source_image_path, img_name), os.path.join(target_image_path, img_name))

def get_img_subset(annotation_file, subcat_names, not_subcat_names):

    # get image ids for images that contain an instance of the subcategories

    coco = COCO(annotation_file)
    subcat_ids = coco.getCatIds(catNms=subcat_names)
    not_subcat_ids = coco.getCatIds(catNms=not_subcat_names)

    # list of image ids belonging to sub categories
    subcat_imgs_ids_list = []
    for cat in subcat_ids:
        subcat_imgs = coco.catToImgs[cat]
        for id in subcat_imgs:
            subcat_imgs_ids_list.append(id)

    subcat_imgs_ids_list = list(set(subcat_imgs_ids_list))

    print('len of sub_cat_imgs ', len(subcat_imgs_ids_list))

    # not_subcat_imgs_ids_list = []
    # for cat in not_subcat_ids:
    #     not_subcat_imgs = coco.catToImgs[cat]
    #     for id in not_subcat_imgs:
    #         not_subcat_imgs_ids_list.append(id)

    #not_subcat_imgs_ids_list = list(set(not_subcat_imgs_ids_list))

    return subcat_imgs_ids_list

if __name__=="__main__":

    # filter the images by category
    train_annotation_file = '/home/riachielsa/mscoco/annotations/instances_train2017.json'
    val_annotation_file = '/home/riachielsa/mscoco/annotations/instances_val2017.json'

    train_coco = COCO(train_annotation_file)
    val_coco = COCO(val_annotation_file)

    # subset of categories size 20
    subcat_names = ['person',  'car',
                    'bird', 'cat', 'dog']

    # the rest
    not_subcat_names = ['traffic light', 'bicycle', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'boat', 'tv',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                        'frisbee', 'horse', 'sheep', 'cow', 'chair', 'couch', 'dining table'
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard', 'tennis racket', 'bottle', 'potted plant', 'bed',
                        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'toilet', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush']

    train_img_set_ids = get_img_subset(train_annotation_file, subcat_names, not_subcat_names)
    val_img_set_ids = get_img_subset(val_annotation_file, subcat_names, not_subcat_names)

    seg_mask_output_paths = ['/home/riachielsa/mscoco/train/masks/',
                             '/home/riachielsa/mscoco/val/masks/']

    annotation_paths = [train_annotation_file, val_annotation_file]

    source_image_paths = ['/home/riachielsa/mscoco/train/train2017/',
                          '/home/riachielsa/mscoco/val/val2017/']

    seg_mask_image_paths = ['/home/riachielsa/mscoco/train/imgs/',
                            '/home/riachielsa/mscoco/val/imgs/']


    img_set_ids = [train_img_set_ids, val_img_set_ids]

    coco_json_to_segmentation(seg_mask_output_paths[0], annotation_paths[0], source_image_paths[0],
                              seg_mask_image_paths[0], img_set_ids[0])
    coco_json_to_segmentation(seg_mask_output_paths[1], annotation_paths[1], source_image_paths[1],
                              seg_mask_image_paths[1], img_set_ids[1])
