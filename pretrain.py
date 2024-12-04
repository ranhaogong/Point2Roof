import argparse
from datetime import datetime

from torch import optim
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
# Experiment setting
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')  # mps for mac
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--log', type=bool, default=True)

# Grouping setting
parser.add_argument('--mask_type', type=str, default='rand')
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--group_size', type=int, default=32)
parser.add_argument('--num_group', type=int, default=4)
# parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_points', type=int, default=128)
# parser.add_argument('--num_output', type=int, default=8192)
parser.add_argument('--num_output', type=int, default=512)

# Transformer setting
parser.add_argument('--trans_dim', type=int, default=384)
parser.add_argument('--drop_path_rate', type=float, default=0.1)

# Encoder setting
parser.add_argument('--encoder_depth', type=int, default=12)
parser.add_argument('--encoder_num_heads', type=int, default=6)
parser.add_argument('--encoder_dims', type=int, default=384)
parser.add_argument('--loss', type=str, default='cdl2')

# sche / optim
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--eta_min', type=float, default=0.000001)
parser.add_argument('--t_max', type=float, default=200)

args = parser.parse_args()
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
save_dir = os.path.join(args.save_dir, 'roof_syn')

print('loading dataset')

train_dset = RoofReconstructionDataset(
    data_path='/data/haoran/DiffPMAE/dataset/SyntheticDataset/file_list',
    pc_path='/data/haoran/DiffPMAE/dataset/SyntheticDataset/pc',
    subset='train',
    n_points=256,
    downsample=True
)

val_dset = RoofReconstructionDataset(
    data_path='/data/haoran/DiffPMAE/dataset/SyntheticDataset/file_list',
    pc_path='/data/haoran/DiffPMAE/dataset/SyntheticDataset/pc',
    subset='test',
    n_points=256,
    downsample=True
)

val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, pin_memory=True)
trn_loader = DataLoader(train_dset, batch_size=args.batch_size, pin_memory=True)