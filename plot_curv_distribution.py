import util
import torch, torchvision
from torchvision import transforms
from models import *
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import numpy as np


def match(D, path, save_item):
    for i in range(len(path)):
        D[path[i]] = save_item[i]
    return    

def match_and_add(D, path, add_item):
    for i in range(len(path)):
        if add_item[i] == False:
            D[path[i]] += 1
    return  

def plot_average(dataset, minlog, maxlog, bins, curv, count, run_times):
    space = (maxlog - minlog) / (bins - 1)
    result = []
    result_number = []

    for j in range(bins-1):
        num_count, total_count = 0, 0
        temp_min, temp_max = (j*space + minlog), (j+1) * space + minlog
        for i in range(len(dataset)):
            if curv[dataset[i][0]] >= temp_min and curv[dataset[i][0]] < temp_max:               
                num_count += 1
                total_count += count[dataset[i][0]]
        
        if num_count != 0: result.append(total_count / (num_count * run_times))
        else: result.append(0)

        if num_count != 0: result_number.append(total_count / (num_count))
        else: result_number.append(0)
        
    return result, result_number 


def main():
    # Set Parser
    parser = argparse.ArgumentParser(description='Curvature experiments')
    parser.add_argument('--noise', default=0.2, type=float, help='noise std')
    parser.add_argument('--model', default='./normal_models/normal_cifar_net_epoch_300.pth', type=str, help='load model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    model_name = args.model
    model = resnet18()
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Curvature Computation
    criterion = torch.nn.CrossEntropyLoss()
    def get_regularized_curvature_for_batch(batch_data, batch_labels, h=1e-3, niter=10, temp=1):
        num_samples = batch_data.shape[0]
        model.eval()
        regr = torch.zeros(num_samples)
        eigs = torch.zeros(num_samples)
        for _ in range(niter):
            v = torch.randint_like(batch_data, high=2).cuda()
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
        
            v = h * (v + 1e-7)
        
            batch_data.requires_grad_()
            outputs_pos = model(batch_data + v)
            outputs_orig = model(batch_data)
            loss_pos = criterion(outputs_pos / temp, batch_labels)
            loss_orig = criterion(outputs_orig / temp, batch_labels)
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data )[0]
        
            regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            eigs += torch.diag(torch.matmul(v.reshape(num_samples,-1), grad_diff.reshape(num_samples,-1).T)).cpu().detach()
            model.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()
        
        eig_estimate = eigs / niter
        curv_estimate = regr / niter
        return eig_estimate, curv_estimate


    # Load Cifar 10 Dataset
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = util.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                              shuffle=True, num_workers=2)

    # Build Curvature Dict
    curvature = defaultdict(int)
    mis_classify = defaultdict(int)
    flag = defaultdict(int)
    
    for i in range(len(dataset)):
        curvature[dataset[i][0]] = 0
        mis_classify[dataset[i][0]] = 0
        flag[dataset[i][0]] = 0
    
    
    # # Calculate each image Curvature
    # for batch_idx, (path, inputs, targets) in enumerate(data_loader):
    #     inputs, targets = inputs.cuda(), targets.cuda()
    #     _, curv_estimate = get_regularized_curvature_for_batch(inputs, targets)
    #     curv_estimate = torch.log(curv_estimate)
    #     match(curvature, path, curv_estimate)

    curvature_name = model_name + '_train_curvature.npy'
    # np.save(curvature_name, curvature)
    # print("Done")
    # Save/Load Curvature Dictionary
    curvature = np.load(curvature_name, allow_pickle='TRUE').item()
    
           
    # Plot Relationship of Curvature and Accuracy
    x, y = [], []
    w_x, w_y = [], []
    for i in range(len(dataset)):   
        if flag[dataset[i][0]] == 0:
            x.append(curvature[dataset[i][0]])
            # y.append(mis_classify[dataset[i][0]])
        else:
            continue
       
    fig, ax1 = plt.subplots()
    minlog, maxlog, point = -20, 0, 21
    bins = np.linspace(minlog, maxlog, point)
    ax1.hist(x, bins, color='blue', alpha=0.4)
    ax1.set_ylim([0, 35000])
    ax1.set_xlabel('Curvature (Log Scale)')
    ax1.set_ylabel('Train Samples', color='b')
    plt.title('Curvature Distribution (Log Scale) for Train Samples')
    figure_name =  './plots/plot_hist.png'
    plt.savefig(figure_name, bbox_inches='tight')
   
    
if __name__ == "__main__":
    main()