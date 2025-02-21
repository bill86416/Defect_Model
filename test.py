'''Train CIFAR10 with PyTorch.'''
import util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from collections import defaultdict
import argparse
from models import *

# This code is for regular evaluation

def test(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (_, inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    return

def main():
   # Set Parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    print('==> Preparing data..')

    # Test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = util.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
   
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print('==> Building model..')
    net = resnet18()
    # net_path = './noise_models/noise_1.0_cifar_net_epoch_300' + '.pth'
    net_path = 'teacher_model.pth'
    net.load_state_dict(torch.load(net_path, weights_only=True))
    net = net.to(device)
    net.eval()
    
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    test(net, testloader, device)
    del net

    
if __name__ == '__main__':
    main()