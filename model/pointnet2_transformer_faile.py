import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import *
from .model_utils import *
from utils import loss_utils
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import PointNetConv, XConv, DynamicEdgeConv
from torch_geometric.nn import fps, global_mean_pool, global_max_pool, knn_graph
from torch_geometric.nn import MLP, PointTransformerConv, knn, radius
from torch_geometric.utils import scatter
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.decimation import decimation_indices
from torch_geometric.utils import softmax

class TransformerBlock(torch.nn.Module):
    """
    Transformer block for PointTransformer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """
    TransitionDown for PointTransformer.
    Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an MLP to augment features dimensionality.
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute k-nearest points for each cluster
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformer(torch.nn.Module):
    """
    PointTransformer.
    """

    def __init__(self, latent_dim, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = 1

        # hidden channels
        dim_model = [32, 64, 128, 256, 512]

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])

        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k))
            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]))
        self.lin = MLP([dim_model[-1], latent_dim], plain_last=False)

    def forward(self, pos, batch=None, x=None):
        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        # x = global_mean_pool(x, batch)

        # MLP blocks
        out = self.lin(x)
        return out
    
class PointNet2(nn.Module):
    def __init__(self, model_cfg, in_channel=3):
        super().__init__()
        self.model_cfg = model_cfg
        self.k = 16  # K值，用于KNN图
        self.latent_dim = 32768  # 最终的特征维度
        self.transformer = PointTransformer(latent_dim=self.latent_dim, k=self.k)
        self.offset_fc = nn.Conv1d(128, 3, 1)
        self.cls_fc = nn.Conv1d(128, 1, 1)
        self.drop = nn.Dropout(0.5)
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

        batch_size, num_points, _ = xyz.size() # 64, 1024
        pos = xyz.view(-1, 3)  # 转换为 (B*N, 3)
        batch = torch.arange(batch_size).repeat_interleave(num_points).to(pos.device)  # Batch索引

        # 初始化特征（假设无额外输入特征）
        features = None

        # 使用 PointTransformer 提取特征
        transformer_output = self.transformer(pos, batch=batch, x=features) # torch.Size([64, 128])

        # 恢复到 (B, N, C) 格式
        transformer_output = transformer_output.view(batch_size, num_points, -1).permute(0, 2, 1)
        x = self.drop(transformer_output)
        pred_offset = self.offset_fc(x).permute(0, 2, 1)
        # BxNx1
        pred_cls = self.cls_fc(x).permute(0, 2, 1)
        if self.training:
            self.train_dict.update({
                'cls_pred': pred_cls,
                'offset_pred': pred_offset
            })
        batch_dict['point_features'] = transformer_output.permute(0, 2, 1)
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