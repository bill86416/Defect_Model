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

# This code adds Normal Distribution Noise after the Convolution Outputs (Training)

# Soft Cross Entropy
def soft_cross_entropy(logits, soft_labels):
    return -(soft_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

# Training
def train(e, teacher_net, student_net, trainloader, optimizer, device, learnable_tensor):
    print('\nEpoch: %d' % e)
    train_loss = 0

    for batch_idx, (_, inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # Teacher Model
        teacher_outputs = teacher_net(inputs)
        # Student Model
        inputs = inputs + learnable_tensor       
        student_outputs = student_net(inputs)
        loss = soft_cross_entropy(student_outputs, teacher_outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
   
    return

    
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
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--noise', default=0.3, type=float, help='noise')
    parser.add_argument('--epoch', default=300, type=int, help='epoch')
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
    

    # Load Teacher Model
    print('==> Building model..')
    teacher_net = resnet18()
    net_path = 'teacher_model.pth'    
    teacher_net.load_state_dict(torch.load(net_path, weights_only=True))
    teacher_net = teacher_net.to(device)
    teacher_net.eval()

    # Load Normal Model
    print('==> Building model..')
    student_net = resnet18()
    net_path = './normal_models/normal_cifar_net_epoch_300' + '.pth' 
    student_net.load_state_dict(torch.load(net_path, weights_only=True))
    student_net = student_net.to(device)
    student_net.eval()

    # Learnable Tensor
    learnable_tensor = nn.Parameter(torch.randn(3, 32, 32))

    # Defect Model (Add Normal Distribution Noise), Alpha denots different std
    def hook(model, input, output):
        mean = 0.0
        alpha = args.noise
        defect = torch.normal(mean, alpha, size=output.size()).to(device)
        output = output + defect 
        return output

    print(device)
    epoch = args.epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([learnable_tensor], lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch) 

    learnable_tensor = learnable_tensor.cuda()
    # Add Forward Hook to All Activation Outputs "after the Convolution Layers"
    teacher_hook_handles = {}
    for name, module in teacher_net.named_modules():
        if isinstance(module, nn.Conv2d): 
            teacher_hook_handles[name] = module.register_forward_hook(hook)

    student_hook_handles = {}
    for name, module in student_net.named_modules():
        if isinstance(module, nn.Conv2d): 
            student_hook_handles[name] = module.register_forward_hook(hook)
            

    if device == 'cuda':
        cudnn.benchmark = True

    
    print(learnable_tensor.requires_grad)
    for e in range(epoch):
        train(e, teacher_net, student_net, trainloader, optimizer, device, learnable_tensor)
        scheduler.step()


    # Save Learnable Patch
    PATH = 'learnable_tensor.pth'
    torch.save(learnable_tensor, PATH)

    del teacher_net
    del student_net
    
if __name__ == '__main__':
    main()