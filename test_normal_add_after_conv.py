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
import copy

# This code adds Normal Distribution Noise after the Convolution Outputs

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
    parser.add_argument('--lr', default=0.0, type=float, help='learning rate')
    parser.add_argument('--noise', default=0.0, type=float, help='noise')
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
    

    # Model

    print('==> Building model..')
    net = resnet18()
    
    # net_path = './cifar_net_mixup' + '.pth'
    # net_path = './noise_models/noise_0.3_cifar_net_epoch_300' + '.pth'
    # net_path = './normal_models/normal_cifar_net_epoch_300' + '.pth'
    net_path = './lognormal_models/thres_curv_noise_0.5_cifar_net_total_epoch_500_epoch_500' + '.pth'
    
    net.load_state_dict(torch.load(net_path, weights_only=True))
    net = net.to(device)
    net.eval()

     # Defect Model (Add Normal Distribution Noise), Alpha denots different std
    def hook(model, input, output):
        mean = 0.0
        alpha = args.noise
        defect = torch.normal(mean, alpha, size=output.size()).to(device)
        output = output + defect 
        return output

    
    # Add Forward Hook to All Activation Outputs "after the Convolution Layers"
    hook_handles = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d): 
            hook_handles[name] = module.register_forward_hook(hook)

    if device == 'cuda':
        cudnn.benchmark = True

    test(net, testloader, device)
    del net

    
if __name__ == '__main__':
    main()