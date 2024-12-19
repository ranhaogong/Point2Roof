import os
import random

# 定义路径
output_dir = "/data/haoran/dataset/building3d/Point2Roof"
train_file = os.path.join(output_dir, "train.txt")
test_file = os.path.join(output_dir, "test.txt")

# 获取所有文件夹路径
building_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

# 打乱数据集
random.shuffle(building_dirs)

# 划分比例
split_ratio = 10
split_index = len(building_dirs) // (split_ratio + 1) * split_ratio

# 划分数据集
train_set = building_dirs[:split_index]
test_set = building_dirs[split_index:]

# 写入文件
def write_to_file(file_path, data):
    with open(file_path, "w") as f:
        for path in data:
            f.write(f"{path}\n")

write_to_file(train_file, train_set)
write_to_file(test_file, test_set)

print(f"Train set saved to: {train_file}")
print(f"Test set saved to: {test_file}")
