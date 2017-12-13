
Code for Honours Thesis (ECSE 498-499) Supervised by Prof. Joseph Vybihal

Code for semantic segmentation can be found in the segmentation folder. 

test_set_imgs.npy: 50 test images stored in numpy array
test_set_masks.npy: corresponding masks stored in numpy array
new_weights.h5: last computed weights of the model in recombinator_cnn.py
recombinator_cnn.json: model saved as json file


create_coco_dataset.py: code for reading mask annotations in RLE format and saving them in png format. 

generate_np_files.py: code for preprocessing images and saving them as numpy array, and for converting png mask files into one-hot encoded tensors and saving them as numpy array. 

class_weights.py: code for computing relative pixel frequencies in the training and validation set 

cnn_multigpu.py: code for training a model based on [1], written to train on a Google Cloud Compute Instance, on 2 GPU's. 

recombinator_cnn.py: code for training a model based on [2], written to train on a Google Cloud Compute Instance, on 2 GPU's.

visualize_filters.py: code for visualizing a few feature maps from the VGG16 network

evaluate_model.py: code for generating IoU, loss function scores, and confusion matrices. Can be run to visualize results on the test set. 
First the image will appear, the the ground truth and third the prediction. 

Depends on Python 3, Tensorflow and Keras for running the model, Numpy for computing scores, Matplotlib and Seaborn for visualization. 

References:

[1] Guanbin Li and Yizhou Yu. Deep contrast learning for salient object detection. In
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
    pages 478–487, 2016

[2] Sina Honari, Jason Yosinski, Pascal Vincent, and Christopher J. Pal. Recombinator
    networks: Learning coarse-to-fine feature aggregation. CoRR, abs/1511.07356, 2015.


