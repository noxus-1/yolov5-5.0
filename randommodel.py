import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

def saverandommodel(model,ema,name):
    ckpt = {'epoch': 0,
            'best_fitness': 0.0,
            'training_results': None,
            'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'ema': None,
            'updates': None,
            'optimizer': None,
            'wandb_id':None,
            }

    # Save last, best and delete
    torch.save(ckpt, name)

if __name__ == '__main__':
    gd = [ round(_gd,2) for _gd in range(3,12)/9 ]
    gw = [ round(_gd,2) for _gd in range(6,15)/12 ]
    import pdb
    pd.set_trace()
    saverandommodel()