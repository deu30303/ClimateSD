import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from network.srresnet import _NetG
from dataset import ClimateDataset, CutBlurClimateDataset
from torchvision import models, transforms, datasets
import torch.utils.model_zoo as model_zoo
import numpy as np
from tqdm import tqdm
from cutmix.utils import CutMixCrossEntropyLoss
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--channels", type=str, default='0,1,2', help="channels to be used")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--gpus", default="1", type=str, help="gpu ids")

# Model adds-on:
parser.add_argument('--position', action='store_true', help='Enable position encoding')
parser.add_argument('--cutblur', action='store_true', help='Enable cutblur')
parser.add_argument('--saliency', action='store_true', help='Enable saliency detection')
parser.add_argument("--piece", type=int, default=1, help="pieces")
parser.add_argument('--second', action='store_true', help='Apply augmentation on second channel only')
parser.add_argument('--first', action='store_true', help='Apply augmentation on first channel only')

# Model hyperparameters
parser.add_argument('--r_factor', type=int, default=2, help="r_factor hyper-parameter")
parser.add_argument('--pos_rfactor', type=int, default=4, help="pos_rfactor hyper-parameter")
parser.add_argument("--pooling", default="mean", type=str, help="mean or max")


def main():

    global opt, model, netContent
    opt = parser.parse_args()
    opt.cuda = True

    opt.seed = 2021
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Channel in use: ", opt.channels)
    channels = [int(x) for x in opt.channels.split(',')]

    print("===> Loading datasets")
    
    mean= np.array([289.8653, 1.8082936e-08, 2459.4797])[channels]
    std= np.array([12.206111, 7.2966614e-08, 4773.867])[channels]

    data_transform = transforms.Compose([
        transforms.Normalize(mean=mean,
                              std=std)
    ])
        
    train_set = ClimateDataset('./data/LR_train.npy', './data/HR_train.npy', transform = data_transform)
    eval_set = ClimateDataset('./data/LR_val.npy', './data/HR_val.npy', transform = data_transform)
    
    if opt.cutblur:
        print("===> Cutblur")
        train_set = CutBlurClimateDataset(train_set, 4, saliency = opt.saliency, first = opt.first, second = opt.second, k = opt.piece)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    eval_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

    print("===> Building model")
    if opt.position:
        print("===> Add position encodding")
    
    model = _NetG(len(channels), use_pos = opt.position, r_factor= opt.r_factor, pos_rfactor=opt.pos_rfactor, pooling=opt.pooling)
    model = torch.nn.DataParallel(model)
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if opt.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        mse = eval_model(eval_data_loader, criterion, model)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, optimizer, model, criterion, epoch):

    channels = [int(x) for x in opt.channels.split(',')]

    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        inputs, targets = Variable(batch[0][:, channels, :, :]), Variable(batch[1][:, channels, :, :], requires_grad=False)
        
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        mse_loss = criterion(outputs, targets)
        
        loss = mse_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (iteration % 300) == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5}, MSE: {:.5}".format(epoch, iteration, len(training_data_loader), loss.item(), mse_loss.item()))

def eval_model(eval_data_loader, criterion, model):
    model.eval()
    mse_sum = 0
    channels = [int(x) for x in opt.channels.split(',')]
    
    with torch.no_grad():
        for iteration, batch in enumerate(eval_data_loader, 1):
            inputs, targets = Variable(batch[0][:, channels, :, :]), Variable(batch[1][:, channels, :, :], requires_grad=False)

            if opt.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            mse_sum += criterion(outputs, targets).item()
        
    print("Eval MSE SUM {}".format(mse_sum))
    return mse_sum
    

def save_checkpoint(model, epoch):
    root = f"checkpoint/srresnet{opt.batchSize}_{opt.channels}"
    if opt.position:
        root += '_pos'
    if opt.cutblur:
        root += '_cutblur'
    if opt.second:
        root += '2'
    if opt.first:
        root += '1'
    if opt.saliency:
        root += '_saliency2_'
        root += str(opt.piece) + 'piece'

    root += '/'
    model_out_path = root + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
