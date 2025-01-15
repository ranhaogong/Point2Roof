import os
import shutil
import random

# 定义文件夹路径
obj_folder = "/data/haoran/dataset/RoofDiffusion/dataset/PoznanRD/roof_obj"
ply_folder = "/data/haoran/dataset/RoofDiffusion/dataset/PoznanRD/roof_point_cloud"
target_folder = "/data/haoran/dataset/RoofDiffusion/dataset/PoznanRD/Point2Roof"

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 获取.obj和.ply文件名列表
obj_files = {os.path.splitext(f)[0] for f in os.listdir(obj_folder) if f.endswith('.obj')}
ply_files = {os.path.splitext(f)[0] for f in os.listdir(ply_folder) if f.endswith('.ply')}

# 找到两种文件的交集
common_files = obj_files.intersection(ply_files)

# 遍历所有同名文件
for file_name in common_files:
    obj_path = os.path.join(obj_folder, f"{file_name}.obj")
    ply_path = os.path.join(ply_folder, f"{file_name}.ply")
    
    # 创建同名文件夹
    file_target_folder = os.path.join(target_folder, file_name)
    os.makedirs(file_target_folder, exist_ok=True)

    # 复制.obj文件并重命名为polygon.obj
    shutil.copy(obj_path, os.path.join(file_target_folder, "polygon.obj"))
    
    # 复制.ply文件并重命名为points.ply
    shutil.copy(ply_path, os.path.join(file_target_folder, "points.ply"))

print(f"成功处理了 {len(common_files)} 对文件！")

# 按照 9:1 划分训练集和测试集
folders = [os.path.join(target_folder, d) for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))]
random.shuffle(folders)

split_index = int(len(folders) * 0.9)
train_folders = folders[:split_index]
test_folders = folders[split_index:]

# 写入 train.txt 和 test.txt
train_file = os.path.join(target_folder, "train.txt")
test_file = os.path.join(target_folder, "test.txt")

with open(train_file, "w") as f:
    for folder in train_folders:
        f.write(folder + "\n")

with open(test_file, "w") as f:
    for folder in test_folders:
        f.write(folder + "\n")

print(f"训练集和测试集已划分并写入到 train.txt 和 test.txt！")