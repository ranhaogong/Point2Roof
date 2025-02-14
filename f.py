import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader_Building3DDataset
from train_utils import train_model
from model.ptv3 import PointTransformerV3, Point
from utils import common_utils
from model import model_utils

num_feat = 3
model = PointTransformerV3(cls_mode=False, in_channels=num_feat, enc_patch_size=(512, 512, 512, 512, 512)).cuda()
patch_size = 1024
batch_size = 6
batch_vals = torch.arange(0, batch_size, step=1)
repeat_vals = torch.tensor([patch_size for i in range(batch_size)])
batch_vals = torch.repeat_interleave(batch_vals, repeat_vals).cuda()
feats = torch.rand((patch_size*batch_size, num_feat)).cuda()
sample_data = {"feat": feats, "batch": batch_vals,
               "coord": feats.cuda(), "grid_size": 0.01}
print("feat: ", feats)
print("feat.shape: ", feats.shape)
print("batch_vals: ", batch_vals)
print("batch_vals.shape: ", batch_vals.shape)
sample_dict = Point(sample_data)
output = model(sample_dict)
print(output['feat'].shape)