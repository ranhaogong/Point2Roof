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
# ply_file = "/data/haoran/Point2Roof/testmydata/0/BID_13250_12127916-b009-46eb-a02b-cbaff4842af7.ply"  # 输入的 .ply 文件路径
# xyz_file = "/data/haoran/Point2Roof/testmydata/0/points.xyz"  # 输出的 .xyz 文件路径
# ply_to_xyz(ply_file, xyz_file)

# import pickle

# # 打开文件并加载变量
# with open('xyz.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)
# loaded_data = loaded_data.cpu()
# pt = loaded_data.numpy()
# # 定义保存路径
# file_path = 'point_cloud.xyz'

# # 保存到 .xyz 文件
# np.savetxt(file_path, pt[0])

def transform_obj(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    vertices = []
    faces = []
    translation = None

    # 解析文件内容
    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            vertices.append([x, y, z])
        elif line.startswith('f '):
            faces.append(line.strip())

    # 确定平移向量
    if vertices:
        first_vertex = vertices[0]
        translation = [-first_vertex[0], -first_vertex[1], -first_vertex[2]]

    # 应用平移到所有顶点
    transformed_vertices = []
    for vertex in vertices:
        transformed_vertex = [
            vertex[0] + translation[0],
            vertex[1] + translation[1],
            vertex[2] + translation[2]
        ]
        transformed_vertices.append(transformed_vertex)

    # 写入新的 OBJ 文件
    with open(output_file, 'w') as file:
        # 写入顶点
        for vertex in transformed_vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        # 写入面
        for face in faces:
            file.write(face + '\n')

# 输入和输出文件路径
input_file = '/data/haoran/Point2Roof/testmydata/0/polygon.obj'
output_file = '/data/haoran/Point2Roof/testmydata/0/polygon_changed.obj'

transform_obj(input_file, output_file)
