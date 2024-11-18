# 定义文件路径
file_path = "test.txt"

# 替换规则
old_path = "/home/lili/ModelReconstruction"
new_path = "/data/haoran/dataset/RoofReconstructionDataset/SyntheticDataset"

# 打开文件并进行替换
with open(file_path, "r") as file:
    lines = file.readlines()  # 读取所有行

# 替换路径
updated_lines = [line.replace(old_path, new_path) for line in lines]

# 将修改后的内容写回文件
with open(file_path, "w") as file:
    file.writelines(updated_lines)

print(f"文件已更新，路径 {old_path} 替换为 {new_path}")