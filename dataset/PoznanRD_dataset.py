import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d

def read_pts(pts_file):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        pts = np.array([f.strip().split(' ') for f in lines], dtype=np.float64)
    return pts

def read_ply(ply_file):
    # '/data/haoran/dataset/RoofDiffusion/dataset/PoznanRD/roof_point_cloud/BID_1538042_0_cc7775c2-28c0-4f80-aebf-cb4704a23bd1.ply'
    # 使用 open3d 读取 ply 文件
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 转换为 numpy 数组
    points = np.asarray(pcd.points, dtype=np.float64)
    return points

def load_obj(obj_file):
    vs, edges = [], set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] == 'v':
            vs.append(vals[1:])
        else:
            obj_data = np.array(vals[1:], dtype=np.int).reshape(-1, 1) - 1
            idx = np.arange(len(obj_data)) - 1
            cur_edge = np.concatenate([obj_data, obj_data[idx]], -1)
            [edges.add(tuple(sorted(e))) for e in cur_edge]
    vs = np.array(vs, dtype=np.float64)
    edges = np.array(list(edges))
    return vs, edges

def extract_before_slash(values):
    """
    从列表中提取每个元素在 `//` 之前的数字部分。

    参数:
        values (list): 包含字符串的列表，格式如 '4//1'。
    
    返回:
        list: 提取后的数字部分。
    """
    return [val.split("//")[0] for val in values]

def load_blender_obj(obj_file):
    vs, edges = [], set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] == '#' or vals[0] == 'o' or vals[0] == 'vn' or vals[0] == 's':
            continue
        if vals[0] == 'v':
            vs.append(vals[1:])
        else:
            extracted_values = extract_before_slash(vals[1:])
            obj_data = np.array(extracted_values, dtype=np.int).reshape(-1, 1) - 1
            idx = np.arange(len(obj_data)) - 1
            cur_edge = np.concatenate([obj_data, obj_data[idx]], -1)
            [edges.add(tuple(sorted(e))) for e in cur_edge]
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


class PoznanRDDataset(Dataset):
    def __init__(self, data_path, training, transform, data_cfg, logger=None):
        with open(data_path, 'r') as f:
            self.file_list = f.readlines()
        self.file_list = [f.strip() for f in self.file_list]
        flist = []
        for l in self.file_list:
            file_name = os.path.basename(l)
            file_name_without_ext = os.path.splitext(file_name)[0]
            flist.append(file_name_without_ext)
            
        self.file_list = flist
        
        self.data_path = data_path.rsplit('/', 1)[0]
        
        self.training = training
        
        self.npoint = data_cfg.NPOINT

        self.transform = transform

        if logger is not None:
            logger.info('Total samples: %d' % len(self))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        
        pc_file_path = self.data_path + '/roof_point_cloud/'
        obj_file_path = self.data_path + '/roof_obj/'
        frame_id = self.file_list[item]
        
        points = read_ply(pc_file_path + frame_id + '.ply')
        points = self.transform(points)

        if len(points) > self.npoint:
            idx = np.random.randint(0, len(points), self.npoint)
        else:
            idx = np.random.randint(0, len(points), self.npoint - len(points))
            idx = np.append(np.arange(0, len(points)), idx)
        np.random.shuffle(idx)


        points = points[idx]


        vectors, edges = load_blender_obj(obj_file_path + frame_id + '.obj')
        min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)


        maxXYZ = np.max(max_pt)
        minXYZ = np.min(min_pt)
        min_pt[:] = minXYZ
        max_pt[:] = maxXYZ

        points = (points - min_pt) / (max_pt - min_pt)
        vectors = (vectors - min_pt) / (max_pt - min_pt)
        points = points.astype(np.float32)
        vectors = vectors.astype(np.float32)
        min_pt = min_pt.astype(np.float32)
        max_pt = max_pt.astype(np.float32)
        pt = np.concatenate(( np.expand_dims(min_pt, 0),  np.expand_dims(max_pt, 0)), axis = 0)
        data_dict = {'points': points, 'vectors': vectors, 'edges': edges, 'frame_id': frame_id, 'minMaxPt': pt}
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
                elif key in ['vectors', 'edges']:
                    max_vec = max([len(x) for x in val])
                    batch_vecs = np.ones((batch_size, max_vec, val[0].shape[-1]), dtype=np.float32) * -1e1
                    for k in range(batch_size):
                        batch_vecs[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_vecs
                elif key in ['frame_id']:
                    ret[key] = val
                elif key in ['minMaxPt']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret




