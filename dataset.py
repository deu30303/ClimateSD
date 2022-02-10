import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from PIL import Image

import random
from cutmix.utils import onehot, rand_bbox
import cv2

def saliency_bbox(img, lam, k=1):
    size = img.size()
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    if k == 1:
        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]
    else:
        random_piece = np.random.randint(0, k)
        indices = (-saliencyMap).argpartition(-k, axis=None)[:-k]
        ix, iy = np.unravel_index(indices, saliencyMap.shape)
        x, y = ix[random_piece], iy[random_piece]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class ClimateDataset(Dataset):
    """Low Resolution dataset."""
    def __init__(self, lr_dir, hr_dir, transform=None, cutmix = False):
        self.lr_data = np.load(lr_dir)
        self.hr_data = np.load(hr_dir)
        self.transform = transform

    def __len__(self):
        return len(self.lr_data)

    def __getitem__(self, idx):
        lr_sample  = self.lr_data[idx]
        hr_sample  = self.hr_data[idx]
        
        lr_sample = lr_sample.transpose((2, 1, 0))
        hr_sample = hr_sample.transpose((2, 1, 0))
        
        lr_sample = torch.from_numpy(lr_sample).unsqueeze(0)
        lr_sample = F.interpolate(lr_sample, (321, 213), mode= 'bicubic').squeeze(0)
        hr_sample = torch.from_numpy(hr_sample)
        
        
        if self.transform:
            lr_sample = self.transform(lr_sample)
            hr_sample = self.transform(hr_sample)

        return lr_sample, hr_sample


class CutBlurClimateDataset(ClimateDataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0, saliency = False, first = False, second = False, k=1):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.saliency = saliency
        self.channel = 0
        if first:
            self.channel = 1
        if second:
            self.channel = 2
        self.k = k
        print("==> Using saliency: ", self.saliency)
        print(k)

    def __getitem__(self, index):
        lr, hr = self.dataset[index]

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)

            if self.saliency:
                bbx1, bby1, bbx2, bby2 = saliency_bbox(lr[1:2], lam, self.k)
            else:
                bbx1, bby1, bbx2, bby2 = rand_bbox(lr.size(), lam)

            if np.random.random() > 0.5:
                # mix inside
                if self.channel == 0:
                    lr[:, bbx1:bbx2, bby1:bby2] = hr[:, bbx1:bbx2, bby1:bby2]
                else:
                    lr[self.channel-1, bbx1:bbx2, bby1:bby2] = hr[self.channel-1, bbx1:bbx2, bby1:bby2]
            else:
                # mix outside
                original_lr, lr = self.dataset[index]
                if self.channel == 0:
                    lr[:, bbx1:bbx2, bby1:bby2] = original_lr[:, bbx1:bbx2, bby1:bby2]
                else:
                    lr[self.channel-1, bbx1:bbx2, bby1:bby2] = original_lr[self.channel-1, bbx1:bbx2, bby1:bby2]

        return lr, hr

    def __len__(self):
        return len(self.dataset)
    
