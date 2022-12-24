
import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import networks.resnet as resnet
import numpy as np
import pandas as pd
import random


torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def get_dataset(name):
    batch_size = 1
    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    testset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def get_model(model:str):
    result = resnet.resnet32()
    result.load_state_dict(torch.load('./trained_models/best_models/resnet32-cifar10/cifar10_resnet32-cifar10_99.pth', map_location=torch.device('cpu')), strict=True)
    result.eval()
    return result

    
def trim_dataset(dataset, net, dataset_path, img_prefix, device):
    i = 0
    csv = pd.read_csv(dataset_path + "/mapping.csv")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if not predicted.eq(targets):
                if os.path.exists(dataset_path + "/" + img_prefix + str(i*2) + ".png"):
                    os.remove(dataset_path + "/" + img_prefix + str(i*2) + ".png")                
                if os.path.exists(dataset_path + "/" + img_prefix + str(i*2+1) + ".png"):
                    os.remove(dataset_path + "/" + img_prefix + str(i*2+1) + ".png")   
                csv = csv[csv["file"] != img_prefix + str(i*2) + ".png"]
                csv = csv[csv["file"] != img_prefix + str(i*2+1) + ".png"]
            i+=1
    csv.to_csv(dataset_path + "/mapping.csv", index=False)


def main(dataset:str, model_name:str, technique:str, img_prefix:str, dataset_path:str):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, _ = get_dataset(dataset)
    model = get_model(model_name)
    model = model.to(device)
    trim_dataset(trainloader, model, dataset_path+"_train", img_prefix, device)
    trim_dataset(testloader, model, dataset_path+"_test", img_prefix, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates an adversarial dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
    parser.add_argument('--model', type=str, default='resnet32-cifar10', help='Model')
    parser.add_argument('--img_prefix', type=str, default='cf10_32_', help='')
    parser.add_argument('--technique', type=str, default='fgsm', help='bb or fgsm')
    parser.add_argument('--dataset_path', type=str, default='./adv_datasets/ciless_pruned/ciless_pruned', help='where to save the dataset')
    args = parser.parse_args()
    main(args.dataset, args.model, args.technique, args.img_prefix, args.dataset_path)

