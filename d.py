import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def pointcloud_to_2d_raster(pointcloud, grid_size=0.1):
    """
    将点云按照z轴投影到2D光栅。
    
    Args:
        pointcloud (np.ndarray): 点云数据 (N, 3)，每行为 (x, y, z)。
        grid_size (float): 光栅单元格大小。

    Returns:
        np.ndarray: 高度二维数组，NaN表示非建筑区域。
        tuple: 光栅的x和y轴范围。
    """
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    # 获取点云边界
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 创建光栅化网格
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # 映射点云到光栅
    grid_x, grid_y = np.digitize(x, x_bins), np.digitize(y, y_bins)
    raster = np.full((len(x_bins), len(y_bins)), np.nan)

    for i in range(len(z)):
        gx, gy = grid_x[i], grid_y[i]
        if np.isnan(raster[gx, gy]) or z[i] > raster[gx, gy]:
            raster[gx, gy] = z[i]

    return raster, (x_bins, y_bins)

def detect_keypoints(raster, window_size=3):
    """
    在2D光栅中检测局部极值点。

    Args:
        raster (np.ndarray): 高度二维数组。
        window_size (int): 滑动窗口大小。

    Returns:
        list: 拐点的索引列表 [(row, col), ...]。
    """
    # 滑动窗口操作
    local_max = ndimage.maximum_filter(raster, size=window_size, mode='constant', cval=np.nan)
    local_min = ndimage.minimum_filter(raster, size=window_size, mode='constant', cval=np.nan)

    # 局部极大值和极小值
    maxima = (raster == local_max) & ~np.isnan(raster)
    minima = (raster == local_min) & ~np.isnan(raster)

    # 合并条件
    keypoints = np.argwhere(maxima | minima)

    return keypoints

# 从文件读取点云
def read_pointcloud(file_path):
    """
    读取点云文件。

    Args:
        file_path (str): 点云文件路径。

    Returns:
        np.ndarray: 点云数据。
    """
    return np.loadtxt(file_path)  # 假设点云文件为txt格式，每行 (x, y, z)

# 示例点云文件路径
file_path = "/path/to/your/pointcloud.txt"
data = read_pointcloud(file_path)

raster, (x_bins, y_bins) = pointcloud_to_2d_raster(data)
keypoints = detect_keypoints(raster)

# 保存可视化结果
plt.imshow(raster.T, origin='lower', extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]), cmap='terrain')
plt.colorbar(label='Height')
plt.scatter(keypoints[:, 1], keypoints[:, 0], color='red', label='Keypoints')
plt.legend()
plt.title('2D Raster and Keypoints')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("raster_keypoints.png")
