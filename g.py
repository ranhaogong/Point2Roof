import os
import numpy as np
import matplotlib.pyplot as plt

# 文件夹路径
folder_path = '/data/haoran/dataset/building3d/roof/Tallinn/test/xyz'

# 获取所有xyz文件
xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]

# 存储每个文件中的点数量
point_counts = []

# 遍历所有xyz文件，统计每个文件中的点数
for file in xyz_files:
    file_path = os.path.join(folder_path, file)
    
    # 读取xyz文件，假设每行表示一个点，且文件内容是空格或制表符分隔
    try:
        points = np.loadtxt(file_path)
        point_count = points.shape[0]  # 点数是行数
        point_counts.append(point_count)
    except Exception as e:
        print(f"无法读取文件 {file}: {e}")

# 创建统计直方图并保存到本地
output_path = '/data/haoran/Point2Roof/point_count_histogram.png'  # 设置保存路径
plt.figure(figsize=(10, 6))
plt.hist(point_counts, bins=20, color='skyblue', edgecolor='black')
plt.title('point_count_histogram')
plt.xlabel('point count')
plt.ylabel('file count')
plt.grid(True)

# 保存直方图到本地
plt.savefig(output_path)

# 关闭图像，防止占用内存
plt.close()
