

import os
import pandas as pd
import matplotlib.pyplot as plt

# # 读取当前目录和子目录的csv文件
path = os.getcwd()

# 创建 image 目录（如果不存在）
image_dir = os.path.join(path, "image")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

file_list = []
for root, dirs, files in os.walk(path+"/csv"):
    for file in files:
        if file.endswith('.csv'):
            file_list.append(os.path.join(root, file))

for file in file_list:
    print(file)
    data = pd.read_csv(file)
    # 获取文件名，不带扩展名
    base_filename = os.path.splitext(os.path.basename(file))[0]

    # 读取位置数据
    pos_data = data[['pos_x', 'pos_y', 'pos_z']]

    # # 创建一个3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D轨迹
    ax.plot(pos_data['pos_x'], pos_data['pos_y'], pos_data['pos_z'])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f"{base_filename}_3D")

    plt.show()

    # 2d轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(pos_data['pos_x'], pos_data['pos_y'])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f"{base_filename}_2D")
    plt.show()