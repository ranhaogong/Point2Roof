import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader_Building3DDataset
from train_utils import train_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils
import torch.optim.lr_scheduler as lr_scheduler


class WarmupScheduler(lr_scheduler._LRScheduler):
    """
    Implements a learning rate warm-up schedule followed by a Cosine Annealing schedule.
    The learning rate will start small and increase linearly for the first `warmup_epochs` epochs.
    After that, it follows a CosineAnnealingLR schedule.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Get the current epoch
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup phase
            warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            return [warmup_lr for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing phase
            cosine_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                        (1 + torch.cos(torch.tensor(epoch - self.warmup_epochs) * torch.pi / (self.max_epochs - self.warmup_epochs)))
            return [cosine_lr for _ in self.optimizer.param_groups]

def get_optimizer_and_scheduler(net, args):
    # AdamW 优化器
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # 设置学习率热身和 Cosine Annealing 调度器
    scheduler = WarmupScheduler(optimizer, warmup_epochs=8, max_epochs=args.epochs, eta_min=1e-6)

    return optimizer, scheduler


def get_scheduler(name, optim, last_epoch):
    if name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5, last_epoch=last_epoch)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20, eta_min=0, last_epoch=last_epoch)
    return scheduler

def get_optimizer(name, net, lr):
    if name == "adam":
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    elif name == "adamw":
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    return opt
    
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--scheduler', type=str, default='steplr', help='scheduler:steplr, cosine')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer:adam, adamw')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max_ckpt_save_num')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--extra_tag', type=str, default='pts6', help='extra tag for this experiment')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    return args, cfg



def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    extra_tag = args.extra_tag if args.extra_tag is not None \
            else 'model-%s' % datetime.datetime.now().strftime('%Y%m%d')
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log.txt'
    logger = common_utils.create_logger(log_file)

    logger.info('**********************Start logging**********************')

    train_loader = build_dataloader_Building3DDataset(args.data_path, args.batch_size, cfg.DATA, training=True, logger=logger, color=cfg.COLOR, nir=cfg.NIR, intensity=cfg.INTENSITY)

    net = RoofNet(cfg.MODEL, color=cfg.COLOR, nir=cfg.NIR, intensity=cfg.INTENSITY)
    net.cuda()
    # optimizer = get_optimizer(args.optimizer, net, args.lr)
    
    optimizer, scheduler = get_optimizer_and_scheduler(net, args)
    start_epoch = it = 0
    last_epoch = -1
    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        print("ckpt_list[-1]: ", ckpt_list[-1])
        ckpt_list.sort(key=os.path.getmtime)
        it, start_epoch = model_utils.load_params_with_optimizer(
            net, ckpt_list[-1], optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1

    # scheduler = get_scheduler(args.scheduler, optimizer, last_epoch=last_epoch)

    net = net.train()
    logger.info('**********************Start training**********************')
    #logger.info(net)
    

    train_model(net, optimizer, train_loader, scheduler, it, start_epoch, args.epochs, ckpt_dir, logger = logger, log_file = log_file, max_ckpt_save_num=args.max_ckpt_save_num)


if __name__ == '__main__':
    main()