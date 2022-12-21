from custom_adversarial_dataset import AdversarialDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import networks.resnet as resnet
import timm
import numpy as np
import pandas as pd

# makes things look nice
from progress_bar import progress_bar


def get_MLP(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 3072),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(3072, 3072),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(3072, 3072),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(3072, 500),
        nn.Linear(500, out_dim)
    )


def get_dataset(name, batch_size=1):
    if name == 'cifar10':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        testset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif name == 'ciless':
        transform = transforms.Compose([transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = AdversarialDataset("adv_datasets/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_train/mapping.csv", "adv_datasets/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_train", transform=transform)
        testset = AdversarialDataset("adv_datasets/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_test/mapping.csv", "adv_datasets/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_test", transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        classes = ('adversrial', 'clean')
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return trainloader, testloader, classes


def save_embedding(img, img_prefix, img_num, path, mapping, target, alternate_embeddings):
    np.save(path + "/" + img_prefix + str(img_num) + ".npy", img.cpu().detach().numpy())
    mapping["file"].append(img_prefix + str(img_num) + ".npy")
    mapping["y"].append(target)
    
    if len(alternate_embeddings) != 0:
        file_names = {}
        for name, embedding in alternate_embeddings.items():
            file_name = img_prefix + str(img_num) + "_" + name + ".npy"
            np.save(path + "/" + file_name, embedding.cpu().detach().numpy())
            file_names[name] = file_name

        mapping["alternate_embeddings"].append(file_names)
    print(mapping)

def get_model(model:str, num_classes:int, load=True):
    if model == 'resnet18':
        result = torchvision.models.resnet18(num_classes=num_classes)
        result.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        result.maxpool = nn.Identity()
    if model == 'resnet32-cifar10':
        result = resnet.resnet32()
        result.load_state_dict(torch.load('./trained_models/best_models/resnet32-cifar10/cifar10_resnet32-cifar10_99.pth', map_location=torch.device('cpu')), strict=True)
    elif model == 'resnet50':
        result = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    elif model == 'vgg11':
        result = torchvision.models.vgg11(weights=None, num_classes=num_classes)
    elif model == 'vgg16':
        result = torchvision.models.vgg16(weights=None, num_classes=num_classes)
    elif model == 'vit':
        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        if load:
            vit.head = nn.Linear(768, 10)
            vit.load_state_dict(torch.load('trained_models/best_models/vit/cifar10_vit_30.pth', map_location=torch.device('cpu')))
        for param in vit.parameters():
            param.requires_grad = False
        vit.head = nn.Identity()
        return vit
    elif model == 'beit':
        beit = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes)
        for param in beit.parameters():
            param.requires_grad = False
        beit.head = get_MLP(768, num_classes)
        # vit.head = nn.Sequential(nn.Linear(768, 1000), nn.Linear(1000, 500), nn.Linear(500, 100), nn.Linear(100, num_classes))
        return beit
    return result



def generate_dataset_fb(net, dataloader, device, img_prefix:str, path:str, alternate_models):
    results = {"file": [], "y": [], "alternate_embeddings":[]}
    eps = []
    img_num = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        embedding = net(inputs)
        alternate_embeddings = dict(map(lambda item: (item[0], item[1](inputs)), alternate_models.items()))
        save_embedding(embedding, img_prefix, img_num, path, results, targets.cpu().item(), alternate_embeddings)
        img_num+=1
    
    df = pd.DataFrame.from_dict(results, orient='columns')
    df.to_csv(path + "/mapping.csv")


def main(dataset:str, model_name:str, img_prefix:str, dataset_path:str, alternate_models):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, classes = get_dataset(dataset)    
    model = get_model(model_name, len(classes))
    model.to(device)
    
    alternate_models_torch = {}
    if alternate_models is not None:
        for alternate_model in alternate_models:
            m = get_model(alternate_model, len(classes))
            m.to(device)
            alternate_models_torch[alternate_model] = m 

    os.makedirs(dataset_path + "_train", exist_ok=True)
    os.makedirs(dataset_path + "_test", exist_ok=True)
    
    model.train()
    for _, m in alternate_models_torch.items():
        m.train()
    generate_dataset_fb(model, trainloader, device, img_prefix, dataset_path + "_train", alternate_models_torch)

    model.eval()
    for _, m in alternate_models_torch.items():
        m.eval()
    generate_dataset_fb(model, testloader, device, img_prefix, dataset_path + "_test", alternate_models_torch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates an adversarial dataset')
    parser.add_argument('--dataset', type=str, default='ciless', help='Dataset')
    parser.add_argument('--model', type=str, default='vit', help='Model')
    parser.add_argument('--img_prefix', type=str, default='ciless_emb_', help='')
    parser.add_argument('--dataset_path', type=str, default='./adv_datasets/ciless_advvit_resnet_output', help='where to save the dataset')
    parser.add_argument('--alternate_models', nargs='+', type=str, default="resnet32-cifar10" help='join multiple model embeddings with original model embeddings')
    args = parser.parse_args()
    main(args.dataset, args.model, args.img_prefix, args.dataset_path, args.alternate_models)


