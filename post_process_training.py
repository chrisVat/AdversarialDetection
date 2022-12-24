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
import time
from tqdm import tqdm

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


class DualMLP(nn.Module):
    def __init__(self, in_dim1, in_dim2, intermediate_dim, out_dim):
        super(DualMLP, self).__init__()
        self.mlp1 = get_MLP(in_dim1, intermediate_dim)
        self.mlp2 = get_MLP(in_dim2, intermediate_dim)
        self.mlp3 = get_MLP(intermediate_dim, out_dim)
    
    def forward(self, x1, x2):
        x1 = nn.Softmax(dim=1)(x1)
        val1 = self.mlp1(x1)
        val2 = self.mlp2(x2)
        val1+=val2
        return self.mlp3(val1)


def get_dataset(name, batch_size):
    if name == 'ciless_cifar_vit_resnet_output':
        transform = None
        trainset = EmbeddingDataset("adv_datasets/ciless_cifar10_vit_resnet_output/adv_datasets/ciless_embeddings_vit_with_mlp_embeddings_train/mapping.csv", "adv_datasets/ciless_cifar10_vit_resnet_output/adv_datasets/ciless_embeddings_vit_with_mlp_embeddings_train", transform=transform)
        testset = EmbeddingDataset("adv_datasets/ciless_cifar10_vit_resnet_output/adv_datasets/ciless_embeddings_vit_with_mlp_embeddings_test/mapping.csv", "adv_datasets/ciless_cifar10_vit_resnet_output/adv_datasets/ciless_embeddings_vit_with_mlp_embeddings_test", transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
        classes = ('adversrial', 'clean')
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return trainloader, testloader, classes


def get_model(model:str, num_classes:int, load=False):
    if model == "dual_mlp":
        dual_mlp = DualMLP(768, 10, 1500, num_classes)
        return dual_mlp
    else:
        raise ValueError('Unknown model: {}'.format(model))


def train(epoch, max_epochs, net, trainloader, optimizer, scheduler, criterion, device, steps_per_update=1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    steps_per_update = 1
    step = 0
    for batch_idx, (vit, targets, cifar) in enumerate(trainloader):
        vit, targets, cifar = vit.to(device), targets.to(device), cifar.to(device)
        outputs = net(vit, cifar)
        loss = criterion(outputs, targets)
        loss.backward()
        step += 1
        if step % steps_per_update == 0:
            optimizer.step()
            optimizer.zero_grad()
            step = 0
        scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(epoch, max_epochs, batch_idx, len(trainloader), 'Loss: %.3f   Acc: %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*correct/total))


def test(epoch, max_epochs, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (vit, targets, cifar) in enumerate(testloader):
            vit, targets, cifar = vit.to(device), targets.to(device), cifar.to(device)
            outputs = net(vit, cifar)
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs, steps_per_epoch=len(trainloader))

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
    parser.add_argument('--dataset', type=str, default='ciless_cifar_vit_resnet_output', help='Dataset to train on')
    parser.add_argument('--model', type=str, default='dual_mlp', help='Model to train')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix to add to model name, to avoid overlapping experiments.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--steps_per_update', type=int, default=1, help='Number of steps per each epoch (For minibatching to save memory)')
    parser.add_argument('--sched_decay', type=float, default=0.5, help='Scheduler Decay')
    parser.add_argument('--step_size', type=int, default=50, help='Step Size For Scheduler')
    parser.add_argument('--load', type=bool, default=True, help='Load best model')
    args = parser.parse_args()
    main(args.dataset, args.model, args.epochs, args.learning_rate, args.batch_size, args.step_size, args.sched_decay, args.output_prefix, args.steps_per_update, args.load)


