# Adversarial Detection

The motivation behind this project is to find new or modify existing techniques to detect adversarial attacks on Computer Vision models.

Our initial goal is to detect adversarial examples effectively and later look into removing adversarial patterns from the images for accurate predictions by the models.

## Research and Development Process

- Our first step was to write setup code for training a model on a given dataset. `train_model.py` file contains the code to train a model and `progress_bar.py` file contains the helper code to display progress bar shown when training a model.

- We will be using cifar-10 dataset and a resnet implementation in `resnet.py` file to evaluate the efficacy of our defenses against adversarial images.

- The next step was to generate adversarial dataset by running adversarial attacks. We use Adversarial Robustness Toolbox and Foolbox, python libraries for Machine Learning Security, enabling developers to detect and evaluate machine learning models against adversarial threats.

- After generating dataset containing adversarial images, the group will now evaluate a set of different techniques that can help defend against such attacks.

### Code Architecture

This section describes the files and important code snippets to help understand the codebase.

#### train_model.py

1. For help on how to run the file, run the following command: `python train_model.py --help`. The command should specify all the arguments the file takes. The arguments allow for configuring the model and the dataset to train the model on, and the training parameters such as epochs, batch size and the learning rate.

1. The training code can be found in `train` method. For training, given that we are training an image classifier, we use the Cross Entropy loss function coupled with Stochastic Gradient Descent optimizer. The code also contains a learning rate scheduler which allows for changing the learning rate after every batch for effective training.

1. `fit_model` method runs training and tests for the model and saves the model which has the best accuracy out of all epochs.

### generate_aversarial_dataset.py

1. For help on how to run the file, run the following command: `python generate_adversarial_dataset.py`. The file arguments allow for configuring the dataset and model for which an attack technique should be used.

1. `generate_dataset_<library>` method is used to run attack for a given model and dataset. For all images in the dataset two versions of it are stored, one modified after the attack and one original.

1. `save_img` method saves the image on the device and also updates the csv file, which stores the mapping of an image with predictions of the model and the actual classification of the image. In the csv file, an additional column is used containing 0 or 1 int value specifying if image is original or modified with attack. The generated images will be stored with `img_prefx + image_number.png` file name.
