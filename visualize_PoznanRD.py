import os
import torch
import torch.nn as nn
import argparse
import datetime
import tqdm
import itertools
import glob
import torch.distributed as dist
import numpy as np
import open3d as o3d
from dataset.data_utils import build_dataloader_PoznanRDDataset
from test_util import test_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from pathlib import Path
from model import model_utils
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--test_tag', type=str, default='pts6', help='extra tag for this experiment')

    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    return args, cfg


def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.test_tag
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    assert output_dir.exists(), '%s does not exist!!!' % str(output_dir)
    ckpt_dir = output_dir / 'ckpt'
    visualize_dir = output_dir / 'results'
    # visualize_dir = "/data/haoran/Point2Roof/testmydata/results"
    output_dir = output_dir / 'test'
    

    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    log_file = visualize_dir / 'log.txt'
    # log_file = "/data/haoran/Point2Roof/testmydata/results/log.txt"
    logger = common_utils.create_logger(log_file)

    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_config_to_file(cfg, logger=logger)

    test_loader = build_dataloader_PoznanRDDataset(args.data_path, args.batch_size, cfg.DATA, training=False, logger=logger)
    net = RoofNet(cfg.MODEL)
    net.cuda()
    net.eval()
    print("ckpt_dir: ", ckpt_dir)
    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    print("ckpt_list: ", ckpt_list)
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        model_utils.load_params(net, ckpt_list[-1], logger=logger)
        print("pth: ", ckpt_list[-1])

    logger.info('**********************Start visualizing**********************')
    # logger.info(net)

    visualize_model(net, test_loader, logger, visualize_dir)

def visualize_model(model, data_loader, logger, visualize_dir):
    dataloader_iter = iter(data_loader)
    with tqdm.trange(0, len(data_loader), desc='test', dynamic_ncols=True) as tbar:
        model.use_edge = True
        for cur_it in tbar:
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            with torch.no_grad():
                batch = model(batch)
                load_data_to_cpu(batch)
            visualize_batch(batch, visualize_dir)
            # break

def visualize_batch(batch, visualize_dir):
    batch_size = batch['batch_size']
    pts_pred, pts_refined, pts_label = batch['keypoint'], batch['refined_keypoint'], batch['vectors']
    edge_pred, edge_label = batch['edge_score'], batch['edges']
    pair_points = batch['pair_points']
    frame_id = batch['frame_id']
    minMaxPt = batch['minMaxPt']
    points = batch['points']
    
    point_pred_score = batch['point_pred_score']
    print("point_pred_score: ", point_pred_score)
    print("point_pred_score.shape: ", point_pred_score.shape)
    print("points: ", points)
    print("points.shape", points.shape)
    points = points[0]
    
    point_pred_score = point_pred_score[0]

    # 初始化颜色数组
    colors = np.ones_like(points) * 255  # 默认白色 (255, 255, 255)

    # 将得分 > 0.5 的点标记为红色 (255, 0, 0)
    colors[point_pred_score > 0.5] = [255, 0, 0]
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D 颜色需要在 [0, 1] 范围内

    ply_name = "colored_points/" + str(frame_id[0]) + ".ply"
    # 保存为 PLY 文件
    o3d.io.write_point_cloud(ply_name, pcd)

    print(f"点云已保存为{ply_name}")
    
    # np.save('point_cloud.npy', points[0])
    # print("点云数据已保存")
    # print("keypoint: ", pts_pred)
    # print("keypoint.shape: ", pts_pred.shape)
    # print("refined_keypoint: ", pts_refined)
    # print("refined_keypoint.shape: ", pts_refined.shape)
    # print("vectors: ", pts_label)
    # print("vectors.shape: ", pts_label.shape)
    # print("edge_pred: ", edge_pred)
    # print("edge_pred.shape: ", edge_pred.shape)
    # print("edge_label: ", edge_label)
    # print("edge_label.shape: ", edge_label.shape)
    # print("pair_points: ", pair_points)
    # print("pair_points.shape: ", pair_points.shape)
    # print("frame_id: ", frame_id)
    # print("minMaxPt: ", minMaxPt)
    # print("minMaxPt.shape: ", minMaxPt[0][0])
    # print("minMaxPt.shape: ", minMaxPt[0][1])
    # print("unnorm", pts_refined * (minMaxPt[0][1] - minMaxPt[0][0]) + minMaxPt[0][0])
    i = 0
    idx = 0
    p_pts = pts_refined[pts_pred[:, 0] == i]
    l_pts = pts_label[i]
    l_pts = l_pts[np.sum(l_pts, -1, keepdims=False) > -2e1]
    vec_a = np.sum(p_pts ** 2, -1)
    vec_b = np.sum(l_pts ** 2, -1)
    dist_matrix = vec_a.reshape(-1, 1) + vec_b.reshape(1, -1) - 2 * np.matmul(p_pts, np.transpose(l_pts))
    dist_matrix = np.sqrt(dist_matrix + 1e-6)
    p_ind, l_ind = linear_sum_assignment(dist_matrix)
    match_edge = list(itertools.combinations(l_ind, 2))
    match_edge = np.array([tuple(sorted(e)) for e in match_edge])
    score = edge_pred[idx:idx+len(match_edge)]
    idx += len(match_edge)
    l_edge = edge_label[i]
    l_edge = l_edge[np.sum(l_edge, -1, keepdims=False) > 0]
    l_edge = [tuple(e) for e in l_edge]
    match_edge = match_edge[score > 0.5]
    print("match_edge: ", match_edge)
    statistics = {'tp_pts': 0, 'num_label_pts': 0, 'num_pred_pts': 0, 'pts_bias': np.zeros(3, np.float),
                      'tp_edges': 0, 'num_label_edges': 0, 'num_pred_edges': 0}
    eval_process(batch, statistics)
    bias = statistics['pts_bias'] / statistics['tp_pts']
    print('pts_recall: %f' % (statistics['tp_pts'] / statistics['num_label_pts']))
    print('pts_precision: %f' % (statistics['tp_pts'] / statistics['num_pred_pts']))
    print('pts_bias: %f, %f, %f' % (bias[0], bias[1], bias[2]))
    print('edge_recall: %f' % (statistics['tp_edges'] / statistics['num_label_edges']))
    print('edge_precision: %f' % (statistics['tp_edges'] / statistics['num_pred_edges']))
    pts_refined = unnorm(pts_refined, minMaxPt)
    p = pts_refined
    write_obj(p, match_edge, frame_id, visualize_dir)

def eval_process(batch, statistics):
    batch_size = batch['batch_size']
    pts_pred, pts_refined, pts_label = batch['keypoint'], batch['refined_keypoint'], batch['vectors']
    edge_pred, edge_label = batch['edge_score'], batch['edges']
    mm_pts = batch['minMaxPt']
    id = batch['frame_id']

    idx = 0
    for i in range(batch_size):
        mm_pt = mm_pts[i]
        minPt = mm_pt[0]
        maxPt = mm_pt[1]
        deltaPt = maxPt - minPt

        p_pts = pts_refined[pts_pred[:, 0] == i]
        l_pts = pts_label[i]
        l_pts = l_pts[np.sum(l_pts, -1, keepdims=False) > -2e1]
        vec_a = np.sum(p_pts ** 2, -1)
        vec_b = np.sum(l_pts ** 2, -1)
        dist_matrix = vec_a.reshape(-1, 1) + vec_b.reshape(1, -1) - 2 * np.matmul(p_pts, np.transpose(l_pts))
        dist_matrix = np.sqrt(dist_matrix + 1e-6)
        p_ind, l_ind = linear_sum_assignment(dist_matrix)
        mask = dist_matrix[p_ind, l_ind] < 0.1   # 0.1
        tp_ind, tl_ind = p_ind[mask], l_ind[mask]
        #dis = np.abs(p_pts[tp_ind] - l_pts[tl_ind])
        dis = np.abs( ((p_pts[tp_ind]*deltaPt) + minPt) - ((l_pts[tl_ind]*deltaPt) + minPt) )


        statistics['tp_pts'] += tp_ind.shape[0]
        statistics['num_label_pts'] += l_pts.shape[0]
        statistics['num_pred_pts'] += p_pts.shape[0]
        statistics['pts_bias'] += np.sum(dis, 0)

        match_edge = list(itertools.combinations(l_ind, 2))
        match_edge = np.array([tuple(sorted(e)) for e in match_edge])
        score = edge_pred[idx:idx+len(match_edge)]
        idx += len(match_edge)
        l_edge = edge_label[i]
        l_edge = l_edge[np.sum(l_edge, -1, keepdims=False) > 0]
        l_edge = [tuple(e) for e in l_edge]
        match_edge = match_edge[score > 0.5]
        tp_edges = np.sum([tuple(e) in l_edge for e in match_edge])
        statistics['tp_edges'] += tp_edges
        statistics['num_label_edges'] += len(l_edge)
        statistics['num_pred_edges'] += match_edge.shape[0]


def unnorm(pts, minMaxPt):
    pts = pts * (minMaxPt[0][1] - minMaxPt[0][0]) + minMaxPt[0][0]
    return pts

def write_obj(pts, faces, frame_id, file_path):
    file_path = file_path / (frame_id[0] + ".obj")
    print(file_path)
    with open(file_path, 'w') as file:
        for point in pts:
            line = f"v {point[0]} {point[1]} {point[2]}\n"
            file.write(line)
        for face in faces:
            face = face + 1
            line = "f " + " ".join(map(str, face)) + "\n"
            file.write(line)
    print(f"点云数据已成功写入文件：{file_path}")

def find_faces(edges):
    # 构建邻接表
    adj_list = defaultdict(set)
    for edge in edges:
        adj_list[edge[0]].add(edge[1])
        adj_list[edge[1]].add(edge[0])

    visited = set()  # 用于存储访问过的边
    faces = []  # 存储所有找到的面

    def dfs(current, start, path):
        """
        深度优先搜索寻找环
        :param current: 当前顶点
        :param start: 起始顶点（用于判断闭环）
        :param path: 当前路径
        """
        for neighbor in adj_list[current]:
            if neighbor == start and len(path) > 2:  # 找到一个闭环
                face = sorted(path)  # 对环的顶点排序，避免重复
                if face not in faces:  # 如果未记录，添加到结果
                    faces.append(face)
            elif neighbor not in path:  # 避免重复访问
                dfs(neighbor, start, path + [neighbor])

    # 遍历所有边，尝试寻找环
    for edge in edges:
        v1, v2 = edge
        if tuple(sorted(edge)) not in visited:
            visited.add(tuple(sorted(edge)))  # 标记边为已访问
            dfs(v1, v1, [v1])  # 从当前边的第一个顶点开始搜索
    all_points = set(point for edge in edges for point in edge)
    faces = filter_faces(faces, all_points)
    return faces


def filter_faces(faces, total_points):
    """
    过滤掉被更大的面完全包含的小面，同时去除包含所有点序号的特殊情况
    :param faces: 原始面列表
    :param total_points: 总点序号集合
    :return: 过滤后的面列表
    """
    faces = sorted(faces, key=len, reverse=True)  # 按面大小从大到小排序
    filtered_faces = []

    for i, face in enumerate(faces):
        # 检查当前面是否包含所有点
        if set(face) == total_points:
            continue  # 跳过包含所有点的特殊面

        # # 检查当前面是否被更大的面包含
        # if not any(set(face).issubset(set(other_face)) for other_face in faces[:i]):
        filtered_faces.append(face)

    return filtered_faces


def write_faces_to_file(faces, file_path):
    """
    将找到的所有面写入文件
    :param faces: 面的列表
    :param file_path: 输出文件路径
    """
    with open(file_path, 'w') as file:
        for face in faces:
            line = "f " + " ".join(map(str, face)) + "\n"  # 格式化为 "f 顶点1 顶点2 ..."
            file.write(line)

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()

def load_data_to_cpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        batch_dict[key] = val.cpu().numpy()
if __name__ == '__main__':
    main()