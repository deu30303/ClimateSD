import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob, warnings
# import scipy.io as sio
# import cv2
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
from metrics import MSE, Pearson, batchify, R2
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Error Calculation")
parser.add_argument('--name', type=str, help="name of the predicted file (.npy)")
parser.add_argument('--filter_season', type=int, default=-1)
parser.add_argument('--data', type=str, default="", help='Dataset')

opt = parser.parse_args()

srres = np.load(opt.name)
data = "data"
if opt.data:
    data += f"_{opt.data}"

if opt.filter_season < 0:
	test_gt = np.load(f'{data}/HR_test.npy')
# test_gt = np.load(f'data/HR_test_{opt.filter_season}.npy')
else:
	test_gt = np.load(f'{data}/monthly/HR_test_{opt.filter_season}.npy')
print(srres.shape, test_gt.shape)

@batchify
def bMSE(x, y): return MSE(x, y)
@batchify
def bPearson(x, y): return Pearson(x, y)
@batchify
def bR2(x, y): return R2(x, y)

rmse1, rmse2 = bMSE(srres, test_gt)
p1, p2 = bPearson(srres, test_gt)
r1, r2 = bR2(srres, test_gt)
print(opt.name)
print('Channel 1, RMSE: {}, R2: {}, Pearson correlation: {}, Mean value: {}'.format(rmse1, r1, p1, np.mean(test_gt[..., 0])))
print('Channel 2, RMSE: {}, R2: {}, Pearson correlation: {}, Mean value: {}'.format(rmse2, r2, p2, np.mean(test_gt[..., 1])))
# print('Channel 3, RMSE: {}, R2: {}, Pearson correlation: {}, Mean value: {}'.format(rmse3, r3, p3, np.mean(test_gt[..., 2])))
