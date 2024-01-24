import argparse
import random
from tqdm import tqdm as tq
import sys
from collections import OrderedDict
import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import grad
import numpy as np
import pandas as pd
from utils import rescale, to_one_hot
from pathlib import Path
import importlib
import json

# 导入模型
cnn = importlib.import_module("pytorch-classification.models.cifar.cnn")
resnext = importlib.import_module("pytorch-classification.models.cifar.resnext")
resnet = importlib.import_module("pytorch-classification.models.cifar.resnet")
alexnet = importlib.import_module("pytorch-classification.models.cifar.alexnet")
densenet = importlib.import_module("pytorch-classification.models.cifar.densenet")

# 超参设置
parser = argparse.ArgumentParser(description='Apply different strategies for MIA to target model.')

parser.add_argument('--seed', type=int, default=0 , help='Set random seed for reproducibility.')
parser.add_argument('--dataset', type=str, default='CIFAR100', help='Which dataset to use for the experiments.')
parser.add_argument('--model_type', type=str, default='AlexNet', help='Model Architecture to attack.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for batched computations.')
parser.add_argument('--data_dir', type=str, default='./data', help='Where to retrieve the dataset.')
parser.add_argument('--trained_dir', type=str, default='./trained_models', help='Where to retrieve trained models.')
parser.add_argument('--output_dir', type=str, default='./', help='Where to store output data.')
parser.add_argument('--dry_run', action='store_true', default=True, help='Test run on 100 samples.')


exp_parameters = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Record time of computation
start_time = time.time()

# Setting seed for reproducibility
seed = exp_parameters.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Loading Datasets
if exp_parameters.dataset == 'MNIST':
    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

if exp_parameters.dataset == 'CIFAR100':
    dataloader = datasets.CIFAR100
    num_classes = 100
elif exp_parameters.dataset == 'CIFAR10':
    dataloader = datasets.CIFAR10
    num_classes = 10
elif exp_parameters.dataset == 'MNIST':
    dataloader = datasets.MNIST
    num_classes = 10

batch_size = exp_parameters.batch_size

data_dir = exp_parameters.data_dir
train_dataset = dataloader(data_dir, train=True, download=True, transform=transform_train)
test_dataset = dataloader(data_dir, train=False, download=True, transform=transform_test)

train_set = DataLoader(train_dataset, batch_size=len(train_dataset) if not exp_parameters.dry_run else 100)
images1 = next(iter(train_set))[0]  # 训练集图片
labels1 = next(iter(train_set))[1]  # 训练集标签

test_set = DataLoader(test_dataset, batch_size=len(test_dataset) if not exp_parameters.dry_run else 100)
images0 = next(iter(test_set))[0] # 测试集图片
labels0 = next(iter(test_set))[1] # 测试集标签

# Preprocessing data
if cuda:
    images1 = images1.cuda()
    images0 = images0.cuda()
    labels1 = labels1.cuda()
    labels0 = labels0.cuda()

if exp_parameters.dataset == 'CIFAR10' or exp_parameters.dataset == 'CIFAR100':
    num_channels = 3
    max1, _ = torch.max(images1.transpose(0, 1).reshape(num_channels, -1), 1)
    min1, _ = torch.min(images1.transpose(0, 1).reshape(num_channels, -1), 1)
    max0, _ = torch.max(images0.transpose(0, 1).reshape(num_channels, -1), 1)
    min0, _ = torch.min(images0.transpose(0, 1).reshape(num_channels, -1), 1)
    Max = torch.max(max0, max1)
    Min = torch.min(min0, min1)
    # 最大最小值归一化
    images1 = rescale(images1, Max, Min)
    images0 = rescale(images0, Max, Min)

# Loading model
model_dir = exp_parameters.trained_dir
if exp_parameters.dataset == 'CIFAR10':
    model_dir = model_dir + '/cifar10'
elif exp_parameters.dataset == 'MNIST':
    model_dir = model_dir + '/MNIST'
model_type = exp_parameters.model_type

if model_type == 'DenseNet':
    model = densenet.DenseNet(Max, Min, depth=190, num_classes=num_classes, growthRate=40)
    checkpoint = torch.load(model_dir + '/densenet-bc-L190-k40/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'ResNext':
    model = resnext.CifarResNeXt(8, 29, num_classes, Max, Min)
    checkpoint = torch.load(model_dir + '/resnext-8x64d/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'ResNet':
    model = resnet.ResNet(164, Max, Min, num_classes=num_classes, block_name='bottleneck')
    checkpoint = torch.load(model_dir + '/resnet-110/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'AlexNet':
    model = alexnet.AlexNet(Max, Min, num_classes=num_classes)
    checkpoint = torch.load(model_dir + '/alexnet/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'CNN':
    model = cnn.CNN(Max, Min)
    checkpoint = torch.load(model_dir + '/cnn/cnn_best_model.pth', map_location=torch.device('cpu'))

Loss = torch.nn.CrossEntropyLoss(reduction='none')

if cuda:
    model.cuda()
    Loss.cuda()

if model_type != 'CNN':
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    counter = 1
    for k, v in state_dict.items():
        if exp_parameters.dataset == 'cifar10':
            if counter < len(state_dict) - 1:
                name = k[:9] + k[16:]
            else:
                name = k
            counter += 1
        else:
            name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
else:
    state_dict = checkpoint
    model.load_state_dict(state_dict)

model.eval()

# gradients norms
gradient_norms0 = []
gradient_norms1 = []

# loss
loss_val0 = None
loss_val1 = None

# label
label_onehot0 = None
label_onehot1 = None

# overall gradients norms
overall_gradient_norms0 = []
overall_gradient_norms1 = []

# Loss attack
with torch.no_grad():
  loss0_ = []
  for i in range(0, images0.shape[0], batch_size):
      loss0_.append(Loss(model(images0[i:i + batch_size]), labels0[i:i + batch_size]).detach())
  loss0 = torch.cat(loss0_, 0)
  loss1_ = []
  for i in range(0, images1.shape[0], batch_size):
      loss1_.append(Loss(model(images1[i:i + batch_size]), labels1[i:i + batch_size]).detach())
  loss1 = torch.cat(loss1_, 0)

print('Loss computation: done')
sys.stdout.flush()
sys.stderr.flush()
loss_val0 = loss0.cpu().data.numpy()
loss_val1 = loss1.cpu().data.numpy()

# Labels in one hot encoding
labels0_1hot = to_one_hot(labels0, num_classes)
labels1_1hot = to_one_hot(labels1, num_classes)

label_onehot0 = labels0_1hot.cpu().data.numpy()
label_onehot1 = labels1_1hot.cpu().data.numpy()

for j in tq(range(images0.shape[0])):
    one_sample = torch.unsqueeze(images0[j], 0)
    loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels0[j], 0))
    grad_ = grad(loss_one_sample[0], model.parameters(), create_graph=True)

    grad_norm_j = []
    aux_grad_list = []
    for i in range(len(grad_)):
        gadient = torch.flatten(grad_[i]).detach()
        aux_grad_list.append(gadient)
        # Statistics on the gradient
        Gmean = torch.mean(gadient)
        Gdiffs = gadient - Gmean
        Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
        Gstd = torch.pow(Gvar, 0.5)
        Gzscores = Gdiffs / Gstd
        Gskews = torch.mean(torch.pow(Gzscores, 3.0))
        Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

        GL1 = torch.norm(gadient, 1)
        GL2 = torch.norm(gadient, 2)
        GLinf = torch.norm(gadient, float('inf'))
        GabsMin = torch.min(torch.abs(gadient))

        grad_norm_j.append([norm.cpu().data.numpy().tolist() for norm in \
                            [-GL1, -GL2, -GLinf, -Gmean, -Gskews, Gkurtoses, GabsMin]])
    gradient_norms0.append(grad_norm_j)

    # compute overall gradients norms
    grad_ = torch.cat(aux_grad_list, 0).detach()
    Gmean = torch.mean(grad_)
    Gdiffs = grad_ - Gmean
    Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
    Gstd = torch.pow(Gvar, 0.5)
    Gzscores = Gdiffs / Gstd
    Gskews = torch.mean(torch.pow(Gzscores, 3.0))
    Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0
    overall_gradient_norms0.append([norm.cpu().data.numpy().tolist() for norm in \
                            [-GL1, -GL2, -GLinf, -Gmean, -Gskews, Gkurtoses, GabsMin]])

    sys.stdout.flush()
    sys.stderr.flush()


for j in tq(range(images1.shape[0])):
    one_sample = torch.unsqueeze(images1[j], 0)
    loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels1[j], 0))
    grad_ = grad(loss_one_sample[0], model.parameters(), create_graph=True)

    grad_norm_j = []
    aux_grad_list = []
    for i in range(len(grad_)):
        gadient = torch.flatten(grad_[i]).detach()
        aux_grad_list.append(gadient)
        # Statistics on the gradient
        Gmean = torch.mean(gadient)
        Gdiffs = gadient - Gmean
        Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
        Gstd = torch.pow(Gvar, 0.5)
        Gzscores = Gdiffs / Gstd
        Gskews = torch.mean(torch.pow(Gzscores, 3.0))
        Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

        GL1 = torch.norm(gadient, 1)
        GL2 = torch.norm(gadient, 2)
        GLinf = torch.norm(gadient, float('inf'))
        GabsMin = torch.min(torch.abs(gadient))

        grad_norm_j.append([norm.cpu().data.numpy().tolist() for norm in \
                            [-GL1, -GL2, -GLinf, -Gmean, -Gskews, Gkurtoses, GabsMin]])
    gradient_norms1.append(grad_norm_j)

    # compute overall gradients norms
    grad_ = torch.cat(aux_grad_list, 0).detach()
    Gmean = torch.mean(grad_)
    Gdiffs = grad_ - Gmean
    Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
    Gstd = torch.pow(Gvar, 0.5)
    Gzscores = Gdiffs / Gstd
    Gskews = torch.mean(torch.pow(Gzscores, 3.0))
    Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

    GL1 = torch.norm(grad_, 1)
    GL2 = torch.norm(grad_, 2)
    GLinf = torch.norm(grad_, float('inf'))
    GabsMin = torch.min(torch.abs(grad_))

    overall_gradient_norms1.append([norm.cpu().data.numpy().tolist() for norm in \
                            [-GL1, -GL2, -GLinf, -Gmean, -Gskews, Gkurtoses, GabsMin]])

    sys.stdout.flush()
    sys.stderr.flush()
print('Gradient norms computation: done')

# Reporting total time of computation

end_time = time.time()

print('Elapsed time of computation in miliseconds: %f' % (end_time - start_time))

# Saving Results
result0 = {
    'gradient_norms': gradient_norms0,
    'loss_val': loss_val0.tolist(),
    'onehot': label_onehot0.tolist(),
    'overall_norms': overall_gradient_norms0
}

result1 = {
    'gradient_norms': gradient_norms1,
    'loss_val': loss_val1.tolist(),
    'onehot': label_onehot1.tolist(),
    'overall_norms': overall_gradient_norms1
}

result_dict = {
    'non-members': result0,
    'members': result1
}

outdir = Path(exp_parameters.output_dir)
results_dir = outdir / 'RawResults/Grad_Norm'
results_dir.mkdir(parents=True, exist_ok=True)

extra_path = exp_parameters.dataset
if exp_parameters.dry_run:
    extra_path = extra_path + '_test'
saved_path = results_dir / f'grad_norms_{model_type}_{extra_path}.json'

with open(saved_path, 'w') as json_file:
    json.dump(result_dict, json_file)
print(f'results saved to {saved_path}')
