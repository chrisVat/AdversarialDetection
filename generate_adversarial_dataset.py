"""
Generate an Adversarial Dataset by running FGSM attack on CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import networks.resnet as resnet
import numpy as np
import foolbox as fb

import pandas as pd
from PIL import Image
import random

from progress_bar import progress_bar


torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def get_dataset(name):
    if name == 'cifar10':
        batch_size = 1
        transform = transforms.Compose([transforms.ToTensor(), 
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        testset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return trainloader, testloader, classes


def get_model(model:str):
    if model == 'resnet32-cifar10':
        result = resnet.resnet32()
        result.load_state_dict(torch.load('./trained_models/best_models/resnet32-cifar10/cifar10_resnet32-cifar10_99.pth', map_location=torch.device('cpu')), strict=True)
        result.eval()
    else:
        raise ValueError('Unknown model: {}'.format(model))
    return result


def save_image(img, img_prefix, img_num, path, mapping, target, actual_class, model_pred):
    uint8_img = (img * 255).astype(np.uint8)
    if len(uint8_img.shape) > 3:
        uint8_img = uint8_img[0]
    
    im = Image.fromarray(uint8_img.transpose(1, 2, 0))
    im.save(path + "/" + img_prefix + str(img_num) + ".png")
    
    mapping["file"].append(img_prefix + str(img_num) + ".png")
    mapping["y"].append(target)
    mapping["class"].append(actual_class)
    mapping["model_pred"].append(model_pred)
    


def get_prediction(net, input_img):
    transform_norm = transforms.Compose([ transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    img = transform_norm(input_img)
    return net(img).argmax(axis=1).item()


def view_prediction(net, input_img, target, prefix='original: '):
    transform_norm = transforms.Compose([ transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    img = transform_norm(input_img)
    test_pred = net(img).argmax(axis=1).item()
    str_prds = prefix + str(test_pred) + " " + str(target.item())
    if test_pred != target:
        print("Incorrect Prediction " + prefix + str_prds)
    else: 
        print("Correct Prediction " + prefix + str_prds)

def generate_dataset_fb(net, dataloader, technique, device, img_prefix:str, path:str):
    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    classifier = fb.PyTorchModel(net, bounds=(0, 1), preprocessing=preprocessing, device=device)
    results = {"file": [], "y": [], "class": [], "model_pred": []}
    eps = []
    if technique == 'bb':
        attack = fb.attacks.brendel_bethge.L2BrendelBethgeAttack(steps=20)
        eps = [0.01, 0.05, 0.1, 0.5]
    elif technique == 'fgsm':
        attack = fb.attacks.fast_gradient_method.L2FastGradientAttack()
        eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    else:
        raise ValueError('Unknown attack: {}'.format(technique))

    img_num = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        raw_advs, clipped_advs, success = attack(classifier, inputs, targets, epsilons=eps)
        success[-1] = True
        for i in range(len(success)):
            if success[i]:
                o_pred = get_prediction(net, inputs)
                a_pred = get_prediction(net, clipped_advs[i])
                save_image(clipped_advs[i].cpu().numpy(), img_prefix, img_num, path, results, targets.cpu().item(), 1, a_pred)
                img_num += 1
                save_image(inputs[0].cpu().numpy(), img_prefix, img_num, path, results, targets.cpu().item(), 0, o_pred)
                img_num += 1
                break
        progress_bar(1, 1, batch_idx, len(dataloader), '%d / %d' % (batch_idx, len(dataloader)))
    df = pd.DataFrame.from_dict(results, orient='columns')
    df.to_csv(path + "/mapping.csv")


def main(dataset:str, model_name:str, technique:str, img_prefix:str, dataset_path:str):
    print("CUDA Available: ", torch.cuda.is_available())
    dataset_path+= "_" + technique
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader, _ = get_dataset(dataset)
    
    model = get_model(model_name)
    model = model.to(device)
    
    os.makedirs(dataset_path + "_train", exist_ok=True)
    os.makedirs(dataset_path + "_test", exist_ok=True)
    
    generate_dataset_fb(model, trainloader, technique, device, img_prefix, dataset_path + "_train")
    generate_dataset_fb(model, testloader, technique, device, img_prefix, dataset_path + "_test")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates an adversarial dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
    parser.add_argument('--model', type=str, default='resnet32-cifar10', help='Model')
    parser.add_argument('--img_prefix', type=str, default='cf10_32_', help='')
    parser.add_argument('--technique', type=str, default='fgsm', help='bb or fgsm')
    parser.add_argument('--dataset_path', type=str, default='./adv_datasets/cifar_10_resnet_32', help='where to save the dataset')
    args = parser.parse_args()
    main(args.dataset, args.model, args.technique, args.img_prefix, args.dataset_path)

