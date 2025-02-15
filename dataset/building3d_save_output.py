import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d

def read_ply(pts_file):
    # 使用 open3d 读取 ply 文件
    pcd = o3d.io.read_point_cloud(pts_file)
    
    # 提取点云中的点数据
    pts = np.asarray(pcd.points)
    # 返回前三列数据（即点云的坐标）
    # print(pts)
    return pts

def read_pts(pts_file):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        # 提取前三列数据
        pts = np.array([line.strip().split(' ')[:3] for line in lines], dtype=np.float64)
    return pts

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
    vs = np.array(vs, dtype=np.float64)
    edges = np.array(list(edges))
    return vs, edges

def writePoints(points, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(points)):
            point = points[i]
            file1.write(str(point[0]))
            file1.write(' ')
            file1.write(str(point[1]))
            file1.write(' ')
            file1.write(str(point[2]))
            file1.write(' ')
            file1.write('\n')


class Building3DDatasetOutput(Dataset):
    def __init__(self, data_path, transform, data_cfg, logger=None):
        with open(data_path, 'r') as f:
            self.file_list = f.readlines()
        self.file_list = [f.strip() for f in self.file_list]
        flist = []
        for l in self.file_list:
             flist.append(l)
        self.file_list = flist

        self.npoint = data_cfg.NPOINT

        self.transform = transform

        if logger is not None:
            logger.info('Total samples: %d' % len(self))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file_path = self.file_list[item]
        frame_id = file_path.split('/')[-1]
        file_form = file_path.split('.')[-1]
        if file_form == 'ply':
            points = read_ply(file_path)
        elif file_form == 'xyz':
            points = read_pts(file_path)
        else:
            print("none support file form")
        if self.transform is not None:
            points = self.transform(points)

        if len(points) > self.npoint:
            idx = np.random.randint(0, len(points), self.npoint)
        else:
            idx = np.random.randint(0, len(points), self.npoint - len(points))
            idx = np.append(np.arange(0, len(points)), idx)
        np.random.shuffle(idx)


        points = points[idx]

        min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)

        maxXYZ = np.max(max_pt)
        minXYZ = np.min(min_pt)
        min_pt[:] = minXYZ
        max_pt[:] = maxXYZ

        centroid = np.mean(points, axis=0)
        points -= centroid
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points /= max_distance
        points = points.astype(np.float32)
        min_pt = min_pt.astype(np.float32)
        max_pt = max_pt.astype(np.float32)
        pt = np.concatenate(( np.expand_dims(min_pt, 0),  np.expand_dims(max_pt, 0)), axis = 0)
        data_dict = {'points': points, 'frame_id': frame_id, 'minMaxPt': pt, 'centroid': centroid, 'max_distance': max_distance}
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key == 'points':
                    ret[key] = np.concatenate(val, axis=0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key in ['frame_id']:
                    ret[key] = val
                elif key in ['minMaxPt']:
                    ret[key] = val
                elif key in ['centroid']:
                    ret[key] = val
                elif key in ['max_distance']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret




