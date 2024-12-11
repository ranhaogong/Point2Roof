# from model.pointnet_util import BallCenterQuery
# import torch
# ball_center_query = BallCenterQuery.apply
# def assign_targets(points, gvs, radius):
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

# def read_xyz(file_path):
#     """
#     读取 .xyz 文件，返回一个包含点坐标的 numpy 数组。
#     :param file_path: .xyz 文件路径
#     :return: numpy 数组，形状为 (n, 3)，每一行代表一个点的 (x, y, z) 坐标
#     """
#     # 使用 numpy 加载文件，假设每行有 3 列数据 (x, y, z)
#     points = np.loadtxt(file_path)
    
#     # 如果需要验证文件格式是否正确（例如，确保每行有三个数字）
#     if points.shape[1] != 3:
#         raise ValueError(f"文件 {file_path} 的数据格式错误，每行应该有 3 个坐标值。")
    
#     return points


# import numpy as np
# points = np.random.rand((64, 1024, 3))
# vs = np.random.rand((64, 24, 3))
# # print("points.shape: ", points.shape)
# # print("points: ", points)
# # print("vs.shape: ", vs.shape)
# # print("vs: ", vs)

# dis, label = assign_targets(points, vs, 0.1)
# print(dis)
# print(label)

# import torch
# print(f"Number of GPUs available: {torch.cuda.device_count()}")
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import os
data_path = '/data/haoran/dataset/building3d/roof/Entry-level/train'
xyz_path = os.path.join(data_path, 'xyz')
print(xyz_path)