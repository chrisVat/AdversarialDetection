"""
Please run this after Generating CILess! We have a bug to solve!
"""

from custom_adversarial_dataset import AdversarialDataset
from custom_embedding_dataset import EmbeddingDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import networks.resnet as resnet
import timm
from tqdm import tqdm

# makes things look nice
from progress_bar import progress_bar

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
testset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

path_to_ciless_train = "adv_datasets/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_train/"
path_to_ciless_test = "adv_datasets/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_test/"


with tqdm(testloader) as tepoch: 
    for batch_idx, (inputs, targets) in enumerate(tepoch):
        torchvision.utils.save_image(inputs, path_to_ciless_test + "cf10_32_" + str(batch_idx*2+1) + ".png")

with tqdm(trainloader) as tepoch: 
    for batch_idx, (inputs, targets) in enumerate(tepoch):
        torchvision.utils.save_image(inputs, path_to_ciless_train + "cf10_32_" + str(batch_idx*2+1) + ".png")

