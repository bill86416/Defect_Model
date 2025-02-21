'''Train CIFAR10 with PyTorch.'''
import util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import os
import argparse
import numpy as np
from models import *

# Mixup
def Mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()  # Shuffle indices for the batch

    mixed_x = lam * x + (1 - lam) * x[index, :]  # Mix the inputs
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# CutMix
def Cutmix(x, y, alpha=1.0):
    """Apply CutMix to the data."""
    # Beta distribution to sample lambda
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()  # random permutation of indices

    # Get the size of the image (assuming square image for simplicity)
    width, height = x.size(2), x.size(3)

    # Randomly select the coordinates of the rectangle to cut
    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)
    bw = int(width * np.sqrt(1 - lam))
    bh = int(height * np.sqrt(1 - lam))

    # Define the coordinates of the cut box
    x1 = max(cx - bw // 2, 0)
    y1 = max(cy - bh // 2, 0)
    x2 = min(cx + bw // 2, width)
    y2 = min(cy + bh // 2, height)

    # Cut and paste the patch
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Mix the labels
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
    

# Loss Function
def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training
def train(e, net, trainloader, optimizer, criterion, device, augmentation):
    print('\nEpoch: %d' % e)
    net.train()
    train_loss = 0

    for batch_idx, (_, inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if augmentation == 'mixup': mixed_data, y_a, y_b, lam = Mixup(inputs, targets)
        else: mixed_data, y_a, y_b, lam = Cutmix(inputs, targets)
            
        optimizer.zero_grad()
        outputs = net(mixed_data)
        
        loss = mix_criterion(criterion, outputs, y_a, y_b, lam)        
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

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    return




def main():
   # Set Parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--aug', default='mixup', type=str, help='augmentation method')
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
    

    # Augmentation Method
    # augmentation = v2.MixUp(num_classes=len(classes))
    # if args.aug == 'mixup': 
    #     augmentation = v2.MixUp(num_classes=len(classes))
    # else:
    #     augmentation = v2.CutMix(num_classes=len(classes))

    augmentation = 'mixup'
    if args.aug == 'mixup': 
        augmentation = 'mixup'
    else:
        augmentation = 'cutmix'
        
    # Model
    print('==> Building model..')
    net = resnet18()
    net = net.to(device)
    
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    print(device)
    epoch = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch) 

    for e in range(epoch):
        train(e, net, trainloader, optimizer, criterion, device, augmentation)
        test(net, testloader, device)
        scheduler.step()

    # Save Model
    folder_name = args.aug + '_models'
    PATH = './cifar_net_self_'+ args.aug + '.pth'
    torch.save(net.state_dict(), PATH)
    
if __name__ == '__main__':
    main()