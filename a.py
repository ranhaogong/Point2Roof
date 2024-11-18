import open3d as o3d
import numpy as np

def ply_to_xyz(ply_file, xyz_file):
    """
    将 .ply 文件转换为 .xyz 文件
    :param ply_file: 输入的 .ply 文件路径
    :param xyz_file: 输出的 .xyz 文件路径
    """
    # 读取 .ply 文件
    mesh = o3d.io.read_triangle_mesh(ply_file)

    # 获取点云的顶点信息
    vertices = np.asarray(mesh.vertices)

    # 将顶点坐标保存为 .xyz 文件
    with open(xyz_file, 'w') as f:
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

    print(f"成功将 {ply_file} 转换为 {xyz_file}")

# 示例使用
ply_file = "/data/haoran/Point2Roof/testmydata/0/BID_13250_12127916-b009-46eb-a02b-cbaff4842af7.ply"  # 输入的 .ply 文件路径
xyz_file = "/data/haoran/Point2Roof/testmydata/0/points.xyz"  # 输出的 .xyz 文件路径
ply_to_xyz(ply_file, xyz_file)