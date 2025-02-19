
import torch
import numpy as np
import os
from model.pointnet_util import BallCenterQuery
ball_center_query = BallCenterQuery.apply

def assign_targets(points, gvs, radius):
    idx = ball_center_query(radius, points, gvs).type(torch.int64)
    idx_count_0 = (idx.eq(-1)).sum().item()  
    idx_count_1 = (idx.eq(1)).sum().item()
    print("idx_count:", idx.size())
    print("idx_count_0:", idx_count_0)
    print("idx_count_1:", idx_count_1)
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

def read_pts(pts_file):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        data = [line.strip().split(' ') for line in lines]
        
    data = np.array(data, dtype=np.float32)
    data = data[:, :3]
    return data

def load_obj(obj_file):
    vs, edges = [], set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] == 'v':
            vs.append(vals[1:])
        elif vals[0] == 'l':
            e = [int(vals[1]) - 1, int(vals[2]) - 1]
            edges.add(tuple(sorted(e)))
    vs = np.array(vs, dtype=np.float32)
    edges = np.array(list(edges))
    return vs, edges

def norm(points, vectors):
    min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
    maxXYZ = np.max(max_pt)
    minXYZ = np.min(min_pt)
    min_pt[:] = minXYZ
    max_pt[:] = maxXYZ
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance
    vectors -= centroid
    vectors /= max_distance
    points = points.astype(np.float32)
    vectors = vectors.astype(np.float32)
    return points, vectors

xyz_path = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz"
w_path = "/data/haoran/dataset/building3d/roof/Tallinn/train/wireframe"
output_path = "/data/haoran/Point2Roof/assign_target_building3d_train"

radius = 1

xyz_files = [os.path.join(xyz_path, f) for f in os.listdir(xyz_path) if f.endswith('.xyz')]
w_files = [os.path.join(w_path, f) for f in os.listdir(w_path) if f.endswith('.obj')]
for i, (xyz, w) in enumerate(zip(xyz_files, w_files)):
    points = read_pts(xyz)
    vs, edges = load_obj(w)
    points, vs = norm(points, vs)
    points = torch.tensor(points)
    points = points.cuda()
    points = points.unsqueeze(0)
    vs = torch.tensor(vs)
    vs = vs.cuda()
    vs = vs.unsqueeze(0)
    offset, cls = assign_targets(points, vs, radius)
    count_0 = (cls.eq(0)).sum().item()  # eq(0) 返回一个布尔型张量，sum 求和，item() 获取数值
    count_1 = (cls.eq(1)).sum().item()
    # print(points)
    # print(vs)
    print("点的总个数:", points.size(1))
    print("0 的个数:", count_0)
    print("1 的个数:", count_1)
    # print(cls)
    break
    




