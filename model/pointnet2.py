import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import *
from .model_utils import *
from utils import loss_utils
from .util import sample_and_group 

class PointNet2(nn.Module):
    def __init__(self, model_cfg, in_channel=3):
        super().__init__()
        self.model_cfg = model_cfg
        self.args = {
            'dropout': 0.5
        }
        self.pct = Pct(self.args, 128)
        self.offset_fc = nn.Conv1d(128, 3, 1)
        self.cls_fc = nn.Conv1d(128, 1, 1)
        self.init_weights()
        self.num_output_feature = 128
        if self.training:
            self.train_dict = {}
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidBCELoss()
            )
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss()
            )
            self.loss_weight = self.model_cfg.LossWeight

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        xyz = batch_dict['points']
        vectors = batch_dict['vectors']
        if self.training:
            offset, cls = self.assign_targets(xyz, vectors, self.model_cfg.PosRadius)
            self.train_dict.update({
                'offset_label': offset,
                'cls_label': cls
            })

        fea = self.pct(xyz)
        pred_offset = self.offset_fc(fea).permute(0, 2, 1)
        # BxNx1
        pred_cls = self.cls_fc(fea).permute(0, 2, 1)
        if self.training:
            self.train_dict.update({
                'cls_pred': pred_cls,
                'offset_pred': pred_offset
            })
        batch_dict['point_features'] = fea
        batch_dict['point_pred_score'] = torch.sigmoid(pred_cls).squeeze(-1)
        batch_dict['point_pred_offset'] = pred_offset * self.model_cfg.PosRadius
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_cls, pred_offset = self.train_dict['cls_pred'], self.train_dict['offset_pred']
        label_cls, label_offset = self.train_dict['cls_label'], self.train_dict['offset_label']
        cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        reg_loss = self.get_reg_loss(pred_offset, label_offset, label_cls, self.loss_weight['reg_weight'])
        loss = cls_loss + reg_loss
        loss_dict.update({
            'pts_cls_loss': cls_loss.item(),
            'pts_offset_loss': reg_loss.item(),
            'pts_loss': loss.item()
        })

        pred_cls = pred_cls.squeeze(-1)
        label_cls = label_cls.squeeze(-1)
        pred_logit = torch.sigmoid(pred_cls)
        pred = torch.where(pred_logit >= 0.5, pred_logit.new_ones(pred_logit.shape), pred_logit.new_zeros(pred_logit.shape))
        acc = torch.sum((pred == label_cls) & (label_cls == 1)).item() / torch.sum(label_cls == 1).item()
        #acc = torch.sum(pred == label_cls).item() / len(label_cls.view(-1))
        disp_dict.update({'pts_acc': acc})
        return loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        batch_size = int(pred.shape[0])
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * weight
        return cls_loss

    def get_reg_loss(self, pred, label, cls_label, weight):
        batch_size = int(pred.shape[0])
        positives = cls_label > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_loss_src = self.reg_loss_func(pred, label, weights=reg_weights)  # [N, M]
        reg_loss = reg_loss_src.sum() / batch_size
        reg_loss = reg_loss * weight
        return reg_loss

    def assign_targets(self, points, gvs, radius):
        idx = ball_center_query(radius, points, gvs).type(torch.int64)
        batch_size = gvs.size()[0]
        idx_add = torch.arange(batch_size).to(idx.device).unsqueeze(-1).repeat(1, idx.shape[-1]) * gvs.shape[1]
        gvs = gvs.view(-1, 3)
        idx_add += idx
        target_points = gvs[idx_add.view(-1)].view(batch_size, -1, 3)
        dis = target_points - points
        dis[idx < 0] = 0
        dis /= radius
        label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device),
                            torch.zeros(idx.shape).to(idx.device))
        return dis, label


class PointNetSAModuleMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlps, use_xyz=True):
        """
        PointNet Set Abstraction Module
        :param npoint: int
        :param radii: list of float, radius in ball_query
        :param nsamples: list of int, number of samples in ball_query
        :param in_channel: int
        :param mlps: list of list of int
        :param use_xyz: bool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        mlps = [[in_channel] + mlp for mlp in mlps]
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            r = radii[i]
            nsample = nsamples[i]
            mlp = mlps[i]
            if use_xyz:
                mlp[0] += 3
            self.groupers.append(QueryAndGroup(r, nsample, use_xyz) if npoint is not None else GroupAll(use_xyz))
            self.mlps.append(Conv2ds(mlp))

    def forward(self, xyz, features, new_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, C1, npoint) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz = xyz.contiguous()
        xyz_flipped = xyz.permute(0, 2, 1)
        if new_xyz is None:
            new_xyz = gather_operation(xyz_flipped, furthest_point_sample(
                xyz, self.npoint, 1.0, 0.0)).permute(0, 2, 1) if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(-1)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNetSAModule(PointNetSAModuleMSG):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, use_xyz=True):
        super().__init__(npoint, [radius], [nsample], in_channel, [mlp], use_xyz)


class PointNetFPModule(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp = Conv2ds([in_channel] + mlp)

    def forward(self, pts1, pts2, fea1, fea2):
        """
        :param pts1: (B, n, 3) 
        :param pts2: (B, m, 3)  n > m
        :param fea1: (B, C1, n)
        :param fea2: (B, C2, m)
        :return:
            new_features: (B, mlp[-1], n)
        """
        if pts2 is not None:
            dist, idx = three_nn(pts1, pts2)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(fea2, idx, weight)
        else:
            interpolated_feats = fea2.expand(*fea2.size()[0:2], pts1.size(1))

        if fea1 is not None:
            new_features = torch.cat([interpolated_feats, fea1], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # _, idx = pointnet_util.knn_query(self.nsample, xyz, new_xyz)
        xyz_trans = xyz.permute(0, 2, 1)
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.permute(0, 2, 1).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.permute(0, 2, 1).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
    

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1, new_xyz)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        xyz = xyz.permute(0, 2, 1)
        xyz = self.pos_xyz(xyz)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        # b, n, c
        x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x