from collections import defaultdict

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

    return faces

# 示例数据
edges = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 1],
    [2, 4],
    [3, 5],
    [5, 1]
]

# 调用函数
faces = find_faces(edges)
print("找到的面:", faces)