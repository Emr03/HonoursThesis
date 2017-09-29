from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

def check_data_generator():

    # we create two instances with the same arguments
    image_gen_args = dict(samplewise_center=True,
                         samplewise_std_normalization=True,
                         rotation_range=0.0,
                         width_shift_range=0,
                         height_shift_range=0,
                         zoom_range=0,
                         rescale=1./255)

    mask_gen_args = dict(samplewise_center=False,
                         samplewise_std_normalization=False,
                         rotation_range=0.0,
                         width_shift_range=0,
                         height_shift_range=0,
                         zoom_range=0,
                         rescale=1./255)

    image_datagen = ImageDataGenerator(**image_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    seed = 10
    image_generator = image_datagen.flow_from_directory(
        # '/home/riachichristina/msra/train/images/',
        '/home/elsa/Desktop/Image_Saliency/msra/test/images/',
        class_mode=None,
        color_mode='rgb',
        target_size=(128, 128),
        batch_size=1,
        seed=seed,
        shuffle=False)

    mask_generator = mask_datagen.flow_from_directory(
        # '/home/riachichristina/msra/train/resized_masks/',
        '/home/elsa/Desktop/Image_Saliency/msra/test/masks/',
        class_mode=None,
        color_mode='grayscale',
        target_size=(32, 32),
        batch_size=1,
        seed=seed,
        shuffle=False)

    print('zipping images')
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    max = np.ones((128, 128, 3))*255

    print('viewing images')
    for i in range(3):
        x, y = next(train_generator)
        #plt.imshow(max - x[0, :, :, :])
        plt.imshow(x[0, :, :, :])
        plt.show()

        plt.imshow(y[0, :, :, 0])
        plt.show()

if __name__=="__main__":
    check_data_generator()