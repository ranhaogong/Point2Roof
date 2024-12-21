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

# import os
# data_path = '/data/haoran/dataset/building3d/roof/Entry-level/train'
# xyz_path = os.path.join(data_path, 'xyz')
# print(xyz_path)

# import pickle
# import torch
# import numpy as np
# import open3d as o3d
# # 加载文件中的变量
# with open('xyz.pkl', 'rb') as f:
#     xyz = pickle.load(f)
#     # print(xyz[0])
    
# with open('my_data.pkl', 'rb') as f:
#     label = pickle.load(f)
#     # print(label[0])
# xyz = xyz[0].cpu().numpy()
# label = label[0].cpu().numpy()
# point_cloud = o3d.geometry.PointCloud()
# # xyz = np.asarray(xyz, dtype=np.float64)
# # 设置点云的坐标
# # 计算每一列的最大值和最小值
# print(xyz)
# x_max, y_max, z_max = np.max(xyz, axis=0)
# x_min, y_min, z_min = np.min(xyz, axis=0)

# # 计算每个坐标轴上的差值
# x_diff = x_max - x_min
# y_diff = y_max - y_min
# z_diff = z_max - z_min

# # 输出最大差值
# print(f"X轴最大差值: {x_diff}")
# print(f"Y轴最大差值: {y_diff}")
# print(f"Z轴最大差值: {z_diff}")
# colors = np.ones_like(xyz) * 255  # 默认白色 (255, 255, 255)
# colors[label > 0] = [255, 0, 0]
    
# # 创建 Open3D 点云对象
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D 颜色需要在 [0, 1] 范围内

# ply_name = "vector_label.ply"
# # 保存为 PLY 文件
# o3d.io.write_point_cloud(ply_name, pcd)

# print(f"点云已保存为{ply_name}")


import os

# 输入文件夹路径和输出文件路径
input_dir = "/data/haoran/dataset/building3d/roof/Tallinn/test/xyz"
output_file = "/data/haoran/dataset/building3d/Point2Roof/test_all.txt"

# 获取输入文件夹中所有文件的绝对路径
file_paths = []
for root, _, files in os.walk(input_dir):
    for file in files:
        file_paths.append(os.path.abspath(os.path.join(root, file)))

# 将路径写入输出文件
with open(output_file, "w") as f:
    for path in file_paths:
        f.write(path + "\n")

len(file_paths)  # 输出文件数量，用于确认