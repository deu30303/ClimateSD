import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob, warnings
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from network.srresnet import _NetG
from dataset import ClimateDataset
from torchvision import models, transforms, datasets
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from metrics import MSE, Pearson, batchify
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--channel", type=int, default=2, help="number of channels to be used")
parser.add_argument('--name', type = str, help='name of the files')
parser.add_argument('--checkpoint', type = str, help='name of the checkpoint dir')

warnings.filterwarnings('ignore')

opt = parser.parse_args()
cuda = torch.cuda.is_available()
channel = opt.channel

from dataset import ClimateDataset
print("===> Loading datasets")
data_transform = transforms.Compose([
        transforms.Normalize(mean=[289.8653, 1.8082936e-08, 2459.4797],
                              std=[12.206111, 7.2966614e-08, 4773.867])
    ])
eval_set = ClimateDataset('./data/LR_test.npy', './data/HR_test.npy', transform = data_transform)
eval_data_loader = DataLoader(dataset=eval_set, num_workers=5, batch_size=4, shuffle=False)

def eval(eval_data_loader, criterion, model):
    mean= np.array([289.8653, 1.8082936e-08, 2459.4797])
    std= np.array([12.206111, 7.2966614e-08, 4773.867])
    inv = transforms.Compose([
        transforms.Normalize(mean=(-mean/std).tolist()[:channel],
                            std=(1.0/std).tolist()[:channel])
    ])

    model.eval()
    
    preds = []    

    for iteration, batch in enumerate(eval_data_loader, 1):
        inputs, targets = Variable(batch[0][:, 0:channel, :, :]), Variable(batch[1][:, 0:channel, :, :], requires_grad=False)
        print(iteration)
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()         

        outputs = model(inputs)
        outputs = inv(outputs)
        outputs = Variable(outputs, requires_grad=False) 
        yp = outputs.permute(0, 3, 2, 1)
        preds.append(yp.cpu())
    return np.concatenate(tuple(preds), axis = 0)

model_path = 'checkpoint/{}/model_epoch_50.pth'.format(opt.checkpoint)
if not cuda:
    model = torch.load(model_path, map_location=torch.device('cpu'))["model"]
else:
    model = torch.load(model_path)["model"]

preds = eval(eval_data_loader, None, model)
np.save(f'output/HR_SRResnet{opt.name}', preds)