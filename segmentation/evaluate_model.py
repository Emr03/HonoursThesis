
from keras.models import Model
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
from recombinator_cnn import MSCNN
import tensorflow as tf

class Evaluation:

    def __init__(self, model, n_classes, test_imgs_file, test_masks_file):

        self.model = model
        self.n_classes = n_classes
        # self.test_imgs = np.load(test_imgs_file)['arr_0']
        # self.test_masks = np.load(test_masks_file)['arr_0']

        self.test_imgs = np.load(test_imgs_file)
        self.test_masks = np.load(test_masks_file)

        self.mIoU = 0

        self.IoU_class = [0 for c in range(n_classes)]
        self.class_count = [0 for c in range(n_classes)]
        self.Acc_class = [0 for c in range(n_classes)]
        self.Prec_class = [0 for c in range(n_classes)]

        self.confusion_matrix = np.zeros((n_classes, n_classes))

        self.categorical_loss = 0

        self.class_names = ['background', 'person', 'car',
                            'bird', 'cat', 'dog']

        #self.class_weights = np.load('class_weights.npy')

    def categorical_crossentropy(self, y_true, y_pred):
        """
        computes the categorical crossentropy over the ground truth mask divided by size of mask
        :param y_true: 4D tensor of shape batch_size, w, h, n_classes
        :param y_pred:
        :return:
        """
        epsilon = 10e-8
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

        L_mask = np.multiply(y_true, np.log(y_pred)) # returns shape batch_size, w, h, n_classes
        L_mask = np.sum(L_mask, axis=(0, 1))

        L_mask = L_mask * self.class_weights[np.newaxis, :]

        self.categorical_loss += -np.sum(L_mask) / (256**2 * self.test_imgs.shape[0])


    def classIoU(self, y_true, y_pred):
        """
        computes the intersection over union for a label-output pair
        :param y_true: ground truth tensor
        :param y_pred: predicted tensor
        :return:
        """

        # change y_pred to a binary mask

        y_pred = np.around(y_pred, decimals=0).astype(np.uint8)
        class_list = []
        # find all the gt classes
        for k in range(y_true.shape[-1]):
            ind = np.where(y_true[:, :, k] > 0)
            if len(ind[0]) > 0:
                class_list.append(k)

        for c in class_list:
            I = np.sum(np.multiply(y_true[:, :, c], y_pred[:, :, c]))
            U = np.sum(np.bitwise_or(y_true[:, :, c], y_pred[:, :, c]))
            IoU = I / U
            self.mIoU += IoU
            self.IoU_class[c] += IoU
            self.class_count[c] += 1




    def conf_matrix(self, y_true, y_pred):
        """
        builds a confusion matrix where entry i, j denotes how many pixels of class i were classified as class j
        :param y_true: ground truth tensor
        :param y_pred: predicted tensor
        :return:
        """
        y_true = np.argmax(y_true, axis=-1).flatten()
        y_pred = np.argmax(y_pred, axis=-1).flatten()

        for k in range(y_true.shape[0]):
            self.confusion_matrix[y_true[k]][y_pred[k]] += 1

    def normalize_conf_matrix(self):
        """
        scales the confusion matrix such that it is row stochastic
        :return:
        """
        for r in range(self.n_classes):
            sum = np.sum(self.confusion_matrix[r, :])
            self.confusion_matrix[r, :] = self.confusion_matrix[r, :] / sum

    def normalize_IoU(self):
        """
        compute mean IoU and mean IoU per class
        :return:
        """
        n_inst = sum(self.class_count)
        self.mIoU /= n_inst
        for c in range(self.n_classes):
                if self.class_count[c] > 0:
                    self.IoU_class[c] /= self.class_count[c]
                else:
                    self.IoU_class[c] = 0


    def evaluate_segmentation(self, visualize, conf_mat_filename):

        # mean rgb values on Imagenet as described by VGG16 people
        mean_pixel = [103.939, 116.779, 123.68]

        mean_img = np.ones((256, 256, 3))
        mean_img[:, :, 0] = mean_pixel[0]
        mean_img[:, :, 1] = mean_pixel[1]
        mean_img[:, :, 2] = mean_pixel[2]

        for idx in range(len(self.test_imgs)):
            img = self.test_imgs[idx]

            # convert image to 4D tensor so the model can read it
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

            output_tensor = self.model.predict(x=img, batch_size=1)[0, :, :, :]
            #prediction = np.argmax(output_tensor, axis=-1).astype(np.uint8)
            prediction = np.around(output_tensor, decimals=0).astype(np.uint8)
            prediction = np.argmax(prediction, axis=-1)
            label = self.test_masks[idx]

            if visualize:
                # change one-hot encoded label to mask
                label_img = np.argmax(label, axis = -1)

                img = img + mean_img
                img = img.astype(np.uint8, copy=False)
                plt.imshow(img[0])
                plt.show()

                sns.heatmap(label_img, cmap=sns.hls_palette(self.n_classes, l=.2, s=.8), cbar=True, vmin=0, vmax=self.n_classes-1)
                plt.show()

                sns.heatmap(prediction, cmap=sns.hls_palette(self.n_classes, l=.2, s=.8), cbar=True, vmin=0, vmax=self.n_classes-1)
                plt.show()

            self.classIoU(label, output_tensor)
            #self.categorical_crossentropy(label, output_tensor)
            self.conf_matrix(label, output_tensor)

        self.normalize_conf_matrix()
        self.normalize_IoU()

        print("mean IoU: ", self.mIoU)
        #print('categorical loss ', self.categorical_loss)

        for c in range(self.n_classes):
            print('IoU of class '+ self.class_names[c] + ": ", self.IoU_class[c])

        conf_mat = sns.heatmap(self.confusion_matrix, annot=True, xticklabels=self.class_names, yticklabels=self.class_names)
        plt.show()
        # conf_mat.savefig(conf_mat_filename)

        #np.save(conf_mat_filename, self.confusion_matrix)

if __name__ == "__main__":

    #matplotlib.use('Agg')

    model = MSCNN()

    end2end = model.load_from_json('recombinator_model.json', 'new_weights.h5')
    eval = Evaluation(model=end2end, n_classes=model.n_classes, test_imgs_file='test_set_imgs.npy',
                      test_masks_file='test_set_masks.npy')

    eval.evaluate_segmentation(visualize=True, conf_mat_filename='test_conf_mat')

    # eval = Evaluation(model=end2end, n_classes=model.n_classes, test_imgs_file='val_imgs.npz',
    #                   test_masks_file='val_masks_full.npz')
    # eval.evaluate_segmentation(visualize=False, conf_mat_filename='val_conf_mat')
    #
    # eval = Evaluation(model=end2end, n_classes=model.n_classes, test_imgs_file='train_imgs.npz',
    #                   test_masks_file='train_masks_full.npz')
    # eval.evaluate_segmentation(visualize=False, conf_mat_filename='train_conf_mat')
