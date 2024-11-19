import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader_PoznanRDDataset
from train_utils import train_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils
def get_scheduler(optim, last_epoch):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5, last_epoch=last_epoch)
    return scheduler


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/haoran/dataset/RoofDiffusion/dataset/PoznanRD', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--extra_tag', type=str, default='pts6', help='extra tag for this experiment')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)

    return args, cfg

args, cfg = parse_config()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
train_loader = build_dataloader_PoznanRDDataset(args.data_path, args.batch_size, cfg.DATA, training=True)
dataloader_iter = iter(train_loader)
batch = next(dataloader_iter)
print(batch)