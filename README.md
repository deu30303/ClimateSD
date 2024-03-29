# Downscaling Earth System Models with Deep Learning #
This repo is the PyTorch codes for "Downscaling  Earth System Models with Deep Learning"
> [**Downscaling Earth System Models with Deep Learning**](https://dl.acm.org/doi/abs/10.1145/3534678.3539031)


## Overall model architecture ##
Geospatial Guided Attention Module
<center><img src="./figure/model_architecture.png"> </center>
Localization Guided Augmentation Module
<center><img src="./figure/aug_architecture.png"> </center>

## Usage ##
### - Training ###
```
usage: main_srresnet.py [-h] [--channels CHANNELS] [--batchSize BATCHSIZE]
                        [--nEpochs NEPOCHS] [--lr LR] [--step STEP] [--cuda]
                        [--start-epoch START_EPOCH] [--gpus GPUS] [--position]
                        [--cutblur] [--saliency] [--piece PIECE] [--second]
                        [--first] [--r_factor R_FACTOR]
                        [--pos_rfactor POS_RFACTOR] [--pooling POOLING]

config

  -h, --help            show this help message and exit
  --channels CHANNELS   channels to be used
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs
  --cuda                Use cuda?
  --start-epoch START_EPOCH
                        Manual epoch number
  --threads THREADS     Number of threads for data loader to use
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --gpus GPUS           gpu ids
  --position            Enable position encoding
  --cutblur             Enable cutblur
  --saliency            Enable saliency detection
  --piece PIECE         pieces
  --second              Apply augmentation on second channel only
  --first               Apply augmentation on first channel only
  --r_factor R_FACTOR   R_FACTOR hyperparameter
  --pos_rfactor POS_RFACTOR
                        POS_RFACTOR hyperparameter
  --pooling POOLING     mean or max
```
### - Evaluation ###
```
usage: evaluation.py [-h] [--channel CHANNEL] [--name NAME]
                     [--checkpoint CHECKPOINT]
optional arguments:
  -h, --help            show this help message and exit
  --channel CHANNEL     number of channels to be used
  --name NAME           name of the files
  --checkpoint CHECKPOINT
                        name of the checkpoint dir
```
### - Test Performance Measurement ###
```
usage: compare.py [-h] [--name NAME] [--filter_season FILTER_SEASON]
                  [--data DATA]
optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the predicted file (.npy)
  --filter_season FILTER_SEASON
  --data DATA           Dataset
```

## Data ##
We provide the data for our experiment. You can download the data using following
[link](http://climatedata.ibs.re.kr/data/cesm-hires)
                     
## Output ##
Currently, we support the output for our model.
| Dataset           | Output | 
|-------------------|---------------|
|2x           | [Download](https://drive.google.com/file/d/1ib97DxM5tTBlWRG_3WB2us4GOXkPkdc7/view?usp=sharing)  | 
|4x           | [Download](https://drive.google.com/file/d/19ANhM2OrdflCB0ak5-wF0uOHW0GVPrte/view?usp=sharing)  |
|8x           | [Download](https://drive.google.com/file/d/1ZIV3I_KUI0fbDp0sRMevu8LJKnauSs0Z/view?usp=sharing)  |

## Citation ##
```
@inproceedings{park2022downscaling,
  title={Downscaling Earth System Models with Deep Learning},
  author={Park, Sungwon and Singh, Karandeep and Nellikkattil, Arjun and Zeller, Elke and Mai, Tung Duong and Cha, Meeyoung},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3733--3742},
  year={2022}
}
```