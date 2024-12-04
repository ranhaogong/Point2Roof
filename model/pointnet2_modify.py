import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import *
from .model_utils import *
from utils import loss_utils
import random
# from timm.models.layers import DropPath, trunc_normal_

class PointNet2(nn.Module):
    def __init__(self, model_cfg, in_channel=3):
        super().__init__()
        self.model_cfg = model_cfg
        self.sa1 = PointNetSAModule(256, 0.1, 16, in_channel, [32, 32, 64])
        self.sa2 = PointNetSAModule(128, 0.2, 16, 64, [64, 64, 128])
        self.sa3 = PointNetSAModule(64, 0.4, 16, 128, [128, 128, 256])
        self.sa4 = PointNetSAModule(16, 0.8, 16, 256, [256, 256, 512])
        self.fp4 = PointNetFPModule(768, [256, 256])
        self.fp3 = PointNetFPModule(384, [256, 256])
        self.fp2 = PointNetFPModule(320, [256, 128])
        self.fp1 = PointNetFPModule(128, [128, 128, 128])
        self.shared_fc = Conv1dBN(128, 128)
        self.drop = nn.Dropout(0.5)
        # self.keypoint_det_net = PointTransformer(
        #     in_channels=64,  # 输入特征维度
        #     enc_depths=[2, 2, 2],  # 编码器层数，可以根据任务复杂度调整
        #     enc_channels=[64, 128, 256],  # 编码器各层通道数
        #     num_heads=[2, 4, 8],  # 每层的注意力头数
        #     mlp_ratio=4.0,
        #     attn_drop=0.1,
        #     proj_drop=0.1
        # )
        self.offset_fc = nn.Conv1d(128, 3, 1)
        self.cls_fc = nn.Conv1d(128, 1, 1)
        self.init_weights()
        self.num_output_feature = 128
        if self.training:
            self.train_dict = {}
            self.add_module(
                'cls_loss_func',
                # FocalLoss()
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
            # offset, cls = self.assign_targets(xyz, vectors, self.model_cfg.PosRadius)
            offset, cls, label_weights = self.assign_targets(xyz, vectors, self.model_cfg.PosRadius)
            self.train_dict.update({
                'offset_label': offset,
                'cls_label': cls,
                'label_weights': label_weights
            })

        fea = xyz
        l0_fea = fea.permute(0, 2, 1)
        l0_xyz = xyz

        l1_xyz, l1_fea = self.sa1(l0_xyz, l0_fea)
        l2_xyz, l2_fea = self.sa2(l1_xyz, l1_fea)
        l3_xyz, l3_fea = self.sa3(l2_xyz, l2_fea)
        l4_xyz, l4_fea = self.sa4(l3_xyz, l3_fea)

        l3_fea = self.fp4(l3_xyz, l4_xyz, l3_fea, l4_fea)
        l2_fea = self.fp3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.fp2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.fp1(l0_xyz, l1_xyz, None, l1_fea)

        x = self.drop(self.shared_fc(l0_fea))
        pred_offset = self.offset_fc(x).permute(0, 2, 1)
        # BxNx1
        pred_cls = self.cls_fc(x).permute(0, 2, 1)
        
        # # 输入点云数据并通过 Transformer 处理
        # point_features = self.keypoint_det_net(batch_dict)

        # neighborhood, center = group_divider(xyz)
        # point_features = self.AE_encoder(neighborhood, center)
        # # # 通过全连接层输出偏移和分类结果
        # pred_offset = self.offset_fc(point_features).permute(0, 2, 1)  # (B, N, 3)
        # pred_cls = self.cls_fc(point_features).permute(0, 2, 1)  # (B, N, 1)

        if self.training:
            self.train_dict.update({
                'cls_pred': pred_cls,
                'offset_pred': pred_offset
            })
        batch_dict['point_features'] = l0_fea.permute(0, 2, 1)
        # batch_dict['point_features'] = point_features['point_features']
        batch_dict['point_pred_score'] = torch.sigmoid(pred_cls).squeeze(-1)
        batch_dict['point_pred_offset'] = pred_offset * self.model_cfg.PosRadius
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_cls, pred_offset = self.train_dict['cls_pred'], self.train_dict['offset_pred']
        label_cls, label_offset = self.train_dict['cls_label'], self.train_dict['offset_label']
        # cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        cls_loss = self.get_cls_loss(pred_cls, label_cls, self.train_dict['label_weights'], self.loss_weight['cls_weight'])
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
    
    
    def get_cls_loss(self, pred, label, label_weights, weight):
        batch_size = int(pred.shape[0])
        positives = label > 0
        negatives = label == 0
        label_weights = torch.ones_like(positives)
        # 更新权重：结合正负样本权重和距离权重
        cls_weights = (negatives * 1.0 + positives * label_weights).float()  # 正样本按距离加权
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        # 计算损失
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        # 乘以损失权重
        cls_loss = cls_loss * weight
        return cls_loss

    # def get_cls_loss(self, pred, label, weight):
    #     batch_size = int(pred.shape[0])
    #     positives = label > 0
    #     negatives = label == 0
    #     cls_weights = (negatives * 1.0 + positives * 1.0).float()
    #     pos_normalizer = positives.sum(1, keepdim=True).float()
    #     cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    #     cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
    #     cls_loss = cls_loss_src.sum() / batch_size

    #     cls_loss = cls_loss * weight
    #     return cls_loss

    # def get_reg_loss(self, pred, label, cls_label, weight):
    #     # 根据 cls_label 动态调整权重
    #     distances = torch.norm(pred - label, dim=-1, keepdim=True)
    #     dynamic_weights = torch.exp(-distances)
    #     reg_weights = (cls_label > 0).float() * dynamic_weights
    #     reg_weights /= torch.clamp(reg_weights.sum(1, keepdim=True), min=1.0)
    #     reg_loss_src = self.reg_loss_func(pred, label, weights=reg_weights)
    #     reg_loss = reg_loss_src.sum() / pred.shape[0]
    #     return reg_loss * weight
    
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
        
        # 计算距离
        dis = target_points - points
        dis[idx < 0] = 0  # 对于无效索引，置为零
        dis_norm = torch.norm(dis, dim=-1, keepdim=True)  # 计算点到最近目标点的距离
        dis /= radius  # 归一化距离

        # 标签：根据索引生成
        label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device), 
                            torch.zeros(idx.shape).to(idx.device))
        
        # 添加距离权重：距离越近，权重越大
        distance_weights = torch.exp(-dis_norm / (0.1 * radius))  # 高斯核函数
        label_weights = label * distance_weights.squeeze(-1)  # 仅对正样本赋权重

        return dis, label, label_weights

    # def assign_targets(self, points, gvs, radius):
    #     idx = ball_center_query(radius, points, gvs).type(torch.int64)
    #     batch_size = gvs.size()[0]
    #     idx_add = torch.arange(batch_size).to(idx.device).unsqueeze(-1).repeat(1, idx.shape[-1]) * gvs.shape[1]
    #     gvs = gvs.view(-1, 3)
    #     idx_add += idx
    #     target_points = gvs[idx_add.view(-1)].view(batch_size, -1, 3)
    #     dis = target_points - points
    #     dis[idx < 0] = 0
    #     dis /= radius
    #     label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device),
    #                         torch.zeros(idx.shape).to(idx.device))
    #     return dis, label


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




# class Encoder(nn.Module):  ## Embedding module
#     def __init__(self, encoder_channel):
#         super().__init__()
#         self.encoder_channel = encoder_channel
#         self.first_conv = nn.Sequential(
#             nn.Conv1d(3, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 256, 1)
#         )
#         self.second_conv = nn.Sequential(
#             nn.Conv1d(512, 512, 1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(512, self.encoder_channel, 1)
#         )

#     def forward(self, point_groups):
#         '''
#             point_groups : B G N 3
#             -----------------
#             feature_global : B G C
#         '''
#         bs, g, n, _ = point_groups.shape
#         point_groups = point_groups.reshape(bs * g, n, 3)
#         # encoder
#         feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
#         feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
#         feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
#         feature = self.second_conv(feature)  # BG 1024 n
#         feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
#         return feature_global.reshape(bs, g, self.encoder_channel)


# class Group(nn.Module):  # FPS + KNN
#     def __init__(self, num_group, group_size):
#         super().__init__()
#         self.num_group = num_group
#         self.group_size = group_size
#         self.knn = KNN(k=self.group_size, transpose_mode=True)

#     def forward(self, xyz):
#         '''
#             input: B N 3
#             ---------------------------
#             output: B G M 3
#             center : B G 3
#         '''
#         batch_size, num_points, _ = xyz.shape
#         # fps the centers out
#         center = misc.fps(xyz, self.num_group)  # B G 3
#         # knn to get the neighborhood
#         _, idx = self.knn(xyz, center)  # B G M
#         assert idx.size(1) == self.num_group
#         assert idx.size(2) == self.group_size
#         idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
#         idx = idx + idx_base
#         idx = idx.view(-1)
#         neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
#         neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
#         # normalize
#         neighborhood = neighborhood - center.unsqueeze(2)
#         return neighborhood, center


# ## Transformers
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)

#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
#         super().__init__()

#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
#             )
#             for i in range(depth)])

#     def forward(self, x, pos):
#         for _, block in enumerate(self.blocks):
#             x = block(x + pos)
#         return x


# # Pretrain model
# class PointTransformer(nn.Module):
#     def __init__(self, config, **kwargs):
#         super().__init__()
#         self.config = config
#         # define the transformer argparse
#         self.trans_dim = config.trans_dim
#         self.depth = config.encoder_depth
#         self.drop_path_rate = config.drop_path_rate
#         self.num_heads = config.encoder_num_heads
#         # embedding
#         self.encoder_dims = config.trans_dim
#         self.encoder = Encoder(encoder_channel=self.encoder_dims)
#         self.mask_type = config.mask_type
#         self.mask_ratio = config.mask_ratio

#         self.pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim),
#         )

#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
#         self.blocks = TransformerEncoder(
#             embed_dim=self.trans_dim,
#             depth=self.depth,
#             drop_path_rate=dpr,
#             num_heads=self.num_heads,
#         )

#         self.norm = nn.LayerNorm(self.trans_dim)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv1d):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def _mask_center_block(self, center, noaug=False):
#         if noaug or self.mask_ratio == 0:
#             return torch.zeros(center.shape[:2]).bool()
#         mask_idx = []
#         for points in center:
#             points = points.unsqueeze(0)
#             index = random.randint(0, points.size(1) - 1)
#             distance_matrix = torch.norm(points[:, index].reshape(1,1,3) - points, p=2, dim=-1)
#             idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
#             ratio = self.mask_ratio
#             mask_num = int(ratio * len(idx))
#             mask = torch.zeros(len(idx))
#             mask[idx[:mask_num]] = 1
#             mask_idx.append(mask.bool())
#         bool_masked_pos = torch.stack(mask_idx).to(center.device)
#         return bool_masked_pos

#     def _mask_center_rand(self, center, noaug=False):
#         '''
#             center : B G 3
#             --------------
#             mask : B G (bool)
#         '''
#         B, G, _ = center.shape
#         # skip the mask
#         if noaug or self.mask_ratio == 0:
#             return torch.zeros(center.shape[:2]).bool()

#         self.num_mask = int(self.mask_ratio * G)

#         overall_mask = np.zeros([B, G])
#         for i in range(B):
#             mask = np.hstack([
#                 np.zeros(G - self.num_mask),
#                 np.ones(self.num_mask),
#             ])
#             np.random.shuffle(mask)
#             overall_mask[i, :] = mask
#         overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

#         return overall_mask.to(center.device)  # B G

#     def forward(self, neighborhood, center, noaug=False):
#         # generate mask
#         # if self.mask_type == 'rand':
#         #     bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
#         # else:
#         #     bool_masked_pos = self._mask_center_block(center, noaug=noaug)

#         group_input_tokens = self.encoder(neighborhood)  # B G C

#         batch_size, seq_len, C = group_input_tokens.size()

#         p = self.pos_embed(center)

#         z = self.blocks(group_input_tokens, p)
#         z = self.norm(z)
#         return z



# class PointTransformer(nn.Module):
#     def __init__(self, in_channels, enc_depths, enc_channels, num_heads, mlp_ratio=4.0, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
#         super(PointTransformer, self).__init__()

#         self.enc_layers = nn.ModuleList()
        
#         # Creating the encoder layers
#         for i in range(len(enc_depths)):
#             layer = nn.ModuleList()
#             for j in range(enc_depths[i]):
#                 layer.append(PointTransformerLayer(in_channels=enc_channels[i], num_heads=num_heads[i], 
#                                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
#                                                    attn_drop=attn_drop, proj_drop=proj_drop))
#             self.enc_layers.append(layer)

#     def forward(self, batch_dict):
#         # Retrieve the point cloud and features
#         features = batch_dict['points']  # (B, N, 3)
         

#         # Pass through each encoder block
#         for layer in self.enc_layers:
#             for transformer_layer in layer:
#                 features = transformer_layer(features)

#         batch_dict['point_features'] = features  # Updated features after transformer layers
#         return batch_dict