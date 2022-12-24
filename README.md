# Adversarial Detection

The motivation behind this project is to find new or modify existing techniques to detect adversarial attacks on Computer Vision models.

Our initial goal is to detect adversarial examples effectively and later look into removing adversarial patterns from the images for accurate predictions by the models.

## Research and Development Process

- Our first step was to write setup code for training a model on a given dataset. `train_model.py` file contains the code to train a model and `progress_bar.py` file contains the helper code to display progress bar shown when training a model.

- We will be using cifar-10 dataset and a resnet implementation in `resnet.py` file to evaluate the efficacy of our defenses against adversarial images.

- The next step was to generate adversarial dataset by running adversarial attacks. We use Adversarial Robustness Toolbox and Foolbox, python libraries for Machine Learning Security, enabling developers to detect and evaluate machine learning models against adversarial threats.

- After generating dataset containing adversarial images, the group will evaluated a set of different techniques that can help defend against such attacks.

- We decided to use ViT model with an MLP binary classifier head for our detection technique, as the robust image embeddings produced by ViTs can help us identiy the adversarial examples.

- Our training approach is outlined in the detail in the research paper.

### Code Architecture

This section describes the files and important code snippets to help understand the codebase. The files are described in the order they were used to generate our adversarial detection technique.

#### train_model.py

1. For help on how to run the file, run the following command: `python train_model.py --help`. The command should specify all the arguments the file takes. The arguments allow for configuring the model and the dataset to train the model on, and the training parameters such as epochs, batch size and the learning rate.

1. The training code can be found in `train` method. For training, given that we are training an image classifier, we use the Cross Entropy loss function coupled with Stochastic Gradient Descent optimizer. The code also contains a learning rate scheduler which allows for changing the learning rate after every batch for effective training.

1. `fit_model` method runs training and tests for the model and saves the model which has the best accuracy out of all epochs.

### generate_aversarial_dataset.py

1. For help on how to run the file, run the following command: `python generate_adversarial_dataset.py --help`. The file arguments allow for configuring the dataset and model for which an attack technique should be used.

1. `generate_dataset_<library>` method is used to run attack for a given model and dataset. For all images in the dataset two versions of it are stored, one modified after the attack and one original. We support FGSM attack on CIFAR dataset which can be found here: https://www.cs.toronto.edu/~kriz/cifar.html

1. `save_img` method saves the image on the device and also updates the csv file, which stores the mapping of an image with predictions of the model and the actual classification of the image. In the csv file, an additional column is used containing 0 or 1 int value specifying if image is original or modified with attack. The generated images will be stored with `img_prefx + image_number.png` file name.

### custom_aversarial_dataset.py

The custom adversarial CILess dataset used by the loader to load the dataset for training and test.

### overwrite_cless.py

Once the CILess dataset is generated, run this file to overwrite the clean samples with CIFAR-10. This is because the images stored by `generate_adversarial_dataset.py` saving the clean images created unwanted perturbations that resulted in model accuracy being affected.

### generate_embedding_dataset.py

1. For help on how to run the file, run the following command: `python generate_embedding_dataset.py --help`. The file arguments allow for configuring the dataset and model for which the resulting embeddings will be saved.

1. This file was used to load the ViT and generate the embeddings for all CILess images so our next step of training the MLP used for ViT head could be expedited. Having the embeddings already generated results in MLP being faster to train reducing our overall epochs required in the next set of training sequence.

### custom_embedding_dataset.py

The custom embedding dataset used by the loader the the embedding dataset for training the test, the main difference from custom adversarial dataset being the type of files being read. The embeddings are stored as numpy arrays vs images.

### vit_fine_tuning.py

1. This is the file where all our ViT training experiments were run from. For help on how to run the file, run the following command: `python vit_fine_tuning.py --help`. The file arguments allow for configuring the dataset and model for which will be fine tuned. The user can specify the learning rate, epochs and batch size for the learning process.

1. The specified dataset will be loaded and a specified model will be fine tuned based on the arguments supplied to the file. We used this file to train the MLP with the ciless embeddings and train the ViT and MLP head jointly on the ciless dataset. The accuracy is then reported after every epoch.

## Other Files used for Experiments

### outlier_detection.py

We experimented running outlier detection on ViT embeddings, however due to the high dimensionality of ViT embeddings, we could not produce optimal results.

### sanity_check.py

We experimented here to test the accuracy of our models against the adversarial dataset.

### post_process_training.py

Derieved from the vit_fine_tuning.py file, this file was used to evaluate the second approach of combining ViT embeddings with original model activations to detect adversarial images. As described in the paper, we failed to establish a confidence in this detection approach and opted for our primary approach instead.
