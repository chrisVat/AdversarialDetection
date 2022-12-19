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


def get_dataset(name, batch_size):
    if name == 'cifar10':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
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
    elif name == 'ciless_vit_embedding':
        transform = None
        trainset = EmbeddingDataset("adv_datasets/ciless_embeddings_vit_pretrain/ciless_embeddings_vit_pretrain_train/mapping.csv", "adv_datasets/ciless_embeddings_vit_pretrain/ciless_embeddings_vit_pretrain_train", transform=transform)
        testset = EmbeddingDataset("adv_datasets/ciless_embeddings_vit_pretrain/ciless_embeddings_vit_pretrain_test/mapping.csv", "adv_datasets/ciless_embeddings_vit_pretrain/ciless_embeddings_vit_pretrain_test", transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        classes = ('adversrial', 'clean')
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return trainloader, testloader, classes


def get_model(model:str, num_classes:int, load=False):
    if model == 'resnet18':
        result = torchvision.models.resnet18(num_classes=num_classes)
        result.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        result.maxpool = nn.Identity()
    if model == 'resnet32-cifar10':
        result = resnet.resnet32()
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
            vit.load_state_dict(torch.load('trained_models/best_models/vit/cifar10_vit_30.pth'))
        for param in vit.parameters():
            param.requires_grad = False
        vit.head = get_MLP(768, num_classes) # nn.Sequential(nn.Linear(768, 1000), nn.Linear(1000, 500), nn.Linear(500, 100), nn.Linear(100, num_classes))
        return vit
    elif model == 'vit_pretrained_mlp':
        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        if load:
            vit.head = nn.Linear(768, 10)
            vit.load_state_dict(torch.load('trained_models/best_models/vit/cifar10_vit_30.pth'))
        vit.head = get_MLP(768, num_classes) # nn.Sequential(nn.Linear(768, 1000), nn.Linear(1000, 500), nn.Linear(500, 100), nn.Linear(100, num_classes))
        if load:
            vit.head.load_state_dict(torch.load('trained_models/best_models/vit_embedding_mlp/ciless_pretrained_vit_embedding_mlp_342.pth'))
        return vit
    elif model == 'beit':
        beit = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes)
        for param in beit.parameters():
            param.requires_grad = False
        beit.head = get_MLP(768, num_classes)
        # vit.head = nn.Sequential(nn.Linear(768, 1000), nn.Linear(1000, 500), nn.Linear(500, 100), nn.Linear(100, num_classes))
        return beit
    elif model == "mlp":
        mlp = get_MLP(768, num_classes)
        if load:
            mlp.load_state_dict(torch.load('trained_models/best_models/vit_embedding_mlp/ciless_pretrained_vit_embedding_mlp_342.pth'))
        return mlp


def train(epoch, max_epochs, net, trainloader, optimizer, scheduler, criterion, device, steps_per_update=1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    steps_per_update = 1
    step = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        # scheduler.step()
        loss = criterion(outputs, targets)
        loss.backward()
        step += 1
        if step % steps_per_update == 0:
            optimizer.step()
            optimizer.zero_grad()
            step = 0
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(epoch, max_epochs, batch_idx, len(trainloader), 'Loss: %.3f   Acc: %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*correct/total))
    scheduler.step()


def test(epoch, max_epochs, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(epoch, max_epochs, batch_idx, len(testloader), 'Loss: %.3f   Acc: %.3f%%'
                         % (test_loss/(batch_idx+1), 100.*correct/total))
    return float(correct)/total


def fit_model(model, trainloader, testloader, device, epochs:int, learning_rate:float, sched_decay:float, step_size:int, save_path:str, steps_per_update:int):
    best_acc = -1
    best_name = ""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, sched_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs, steps_per_epoch=len(trainloader))
    # scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):
        train(epoch, epochs, model, trainloader, optimizer, scheduler, criterion, device, steps_per_update)
        acc = test(epoch, epochs, model, testloader, criterion, device)
        if acc > best_acc:
            if best_name != "":
                os.remove(best_name)
            best_acc = acc
            best_name = save_path + "_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), best_name)
    f = open(save_path + "_best.txt", "w")
    f.write(str(best_acc))
    f.close()
    return best_name, best_acc


def main(dataset:str, model_name:str, epochs:int, learning_rate:float, batch_size:int, step_size:int, sched_decay:float, output_prefix:str, steps_per_update:int, load:bool):
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, classes = get_dataset(dataset, batch_size)    
    model = get_model(model_name, len(classes), load)
    model.to(device)
    os.makedirs("trained_models/" + dataset + "/" + model_name +"/", exist_ok=True)
    best_name, best_accuracy = fit_model(model, trainloader, testloader, device, epochs, learning_rate, sched_decay, step_size, "trained_models/" + dataset + "/" + model_name + "/" + output_prefix + dataset + "_" + model_name, steps_per_update)
    print("Training complete: " + best_name + " with accuracy: " + str(round(best_accuracy, 4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--dataset', type=str, default='ciless', help='Dataset to train on')
    parser.add_argument('--model', type=str, default='vit_pretrained_mlp', help='Model to train')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix to add to model name, to avoid overlapping experiments.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--steps_per_update', type=int, default=4, help='Number of steps per each epoch (For minibatching to save memory)')
    parser.add_argument('--sched_decay', type=float, default=0.5, help='Scheduler Decay')
    parser.add_argument('--step_size', type=int, default=30, help='Step Size For Scheduler')
    parser.add_argument('--load', type=bool, default=True, help='Load best model')
    args = parser.parse_args()
    main(args.dataset, args.model, args.epochs, args.learning_rate, args.batch_size, args.step_size, args.sched_decay, args.output_prefix, args.steps_per_update, args.load)


