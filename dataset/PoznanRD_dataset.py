import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d

def read_ply(pts_file):
    point_cloud = o3d.io.read_point_cloud(pts_file)
    pts = np.asarray(point_cloud.points, dtype=np.float64)
    return pts[:, :3]


def load_obj(obj_file):
    vs, edges = [], set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] == 'v':
            vs.append(vals[1:])
        elif vals[0] == 'f':
            split_vals = [int(x.split('//')[0]) for x in vals[1:]]
            obj_data = np.array(split_vals, dtype=np.int).reshape(-1, 1) - 1
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
        points = read_ply(file_path + '/points.ply')
        points = self.transform(points)

        if len(points) > self.npoint:
            idx = np.random.randint(0, len(points), self.npoint)
        else:
            idx = np.random.randint(0, len(points), self.npoint - len(points))
            idx = np.append(np.arange(0, len(points)), idx)
        np.random.shuffle(idx)


        points = points[idx]


        vectors, edges = load_obj(self.file_list[item] + '/polygon.obj')
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
        min_pt = min_pt.astype(np.float32)
        max_pt = max_pt.astype(np.float32)
        pt = np.concatenate(( np.expand_dims(min_pt, 0),  np.expand_dims(max_pt, 0)), axis = 0)
        data_dict = {'points': points, 'vectors': vectors, 'edges': edges, 'frame_id': frame_id, 'minMaxPt': pt, 'centroid': centroid, 'max_distance': max_distance}
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




