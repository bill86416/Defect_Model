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
import numpy as np
from models import *


            
# Add LogNormal Noise
def add_noise(inputs):
    noise = torch.exp(torch.normal(mean=0.0, std=noise_ratio, size=inputs.size())).cuda()
    return inputs * noise

# Add Noise to Batch
def add_noise_to_batch(path, inputs):
    noise_inputs = torch.zeros_like(inputs)
    
    for i in range(len(inputs)):
        # With Samples higher than Curvature Threshold
        if sorted_curvature[path[i]] > high_curvature_threshold:
            noise_inputs[i,:,:,:] = add_noise(inputs[i,:,:,:]).clone()   
            
        else:
            noise_inputs[i,:,:,:] = inputs[i,:,:,:].clone() 
        
    return noise_inputs
    
# Training
def train(e, net, trainloader, optimizer, criterion, device):
    print('\nEpoch: %d' % e)
    net.train()
    train_loss = 0

    for batch_idx, (path, inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Add Noise
        noise_inputs = add_noise_to_batch(path, inputs)
        optimizer.zero_grad()
        outputs = net(noise_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
   
    return


# Evaluation
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

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    return

def main():
    # Get Curvature from pretrained model
    global sorted_curvature
    curvature = np.load('./normal_models/nest_cifar_net_epoch_300.pth_train_curvature.npy',allow_pickle='TRUE').item()   
    sorted_curvature = dict(sorted(curvature.items(), key=lambda kv: kv[1]))
    
   # Set Parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--noise', default=0.5, type=float, help='noise')
    parser.add_argument('--epoch', default=500, type=int, help='epoch')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    print('==> Preparing data..')

    # Train
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    # Test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = util.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    
    testset = util.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
   
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Set high curvature threshold
    global high_curvature_threshold
    high_curvature_threshold = -12.0

    global noise_ratio
    noise_ratio = args.noise


    # Model
    print('==> Building model..')
    net = resnet18()
    net = net.to(device)
    
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    print(device)
    epoch = args.epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch) 

    for e in range(epoch):
        train(e, net, trainloader, optimizer, criterion, device)
        scheduler.step()
        test(net, testloader, device)  
        
    # Save Model
    PATH = './lognormal_models/thres_curv_noise_' + str(noise_ratio) + '_cifar_net_' + 'total_epoch_' + str(args.epoch) + '.pth'
    torch.save(net.state_dict(), PATH)

    
if __name__ == '__main__':
    main()