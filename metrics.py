import math
import torch
import numpy as np
import time, math, glob
import argparse, os
import torch


# Individual 
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(torch.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def MSE(pred, gt):
    diff = (pred - gt)**2
    return math.sqrt(np.mean(diff[..., 0])), math.sqrt(np.mean(diff[..., 1]))

def Pearson(high, low):
    results = []
    for i in range(3):
        x = high[..., i]
        y = low[..., i]
        r = np.sum((x-np.mean(x))*(y-np.mean(y)))/np.sqrt(np.sum((x-np.mean(x))**2)*np.sum((y-np.mean(y))**2))
        results.append(r)
    return tuple(results)

def R2(pred, gt):
    size = gt.shape[0]*gt.shape[1]
    residual = np.mean((pred - gt)**2, axis=(0, 1))
    total = [np.var(np.array(gt[..., i])) for i in range(3)]
    # print(residual, total)
    return [np.mean(1 - residual[i]/total[i]) for i in range(3)]

# Batch
def batchify(criterion):
    def batch_criterion(xs, ys):
        channel1 = []
        channel2 = []
        # channel3 = []
        for x, y in zip(xs, ys):
            score = criterion(x, y)
            channel1.append(score[0])
            channel2.append(score[1])
            # channel3.append(score[2])
        return np.mean(channel1), np.mean(channel2)
    return batch_criterion
