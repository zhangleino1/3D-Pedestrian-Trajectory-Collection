import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取当前目录和子目录的csv文件
path = os.getcwd()

# 创建 image 目录（如果不存在）
image_dir = os.path.join(path, "image")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

file_list = []
for root, dirs, files in os.walk(path + "/csv"):
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
    time_data = data['timestamp'] / 1000  # 将时间戳从毫秒转换为秒
    acce_glob_x = data['acce_glob_x']
    acce_glob_y = data['acce_glob_y']
    acce_glob_z = data['acce_glob_z']
    yaw_degrees = data['yaw_degrees']
    # 用插值法或其他方法填充NaN值
    time_data = time_data.fillna(method='ffill')
    acce_glob_x = acce_glob_x.fillna(0)  # 可以选择用0填充，也可以用前后数据填充
    acce_glob_y = acce_glob_y.fillna(0)
    acce_glob_z = acce_glob_z.fillna(0)
    yaw_degrees = yaw_degrees.fillna(0)

    # 判断是否含义nan
    print("pose_data contains NaN:", np.isnan(acce_glob_x).any())
    print("timestamps contains NaN:", np.isnan(acce_glob_y).any())
    print("new_timestamps contains NaN:", np.isnan(yaw_degrees).any())
        

    # 创建一个图形窗口，包含4个子图
    fig = plt.figure(figsize=(15, 20))

    # 子图1: 3D 轨迹
    ax1 = fig.add_subplot(221, projection='3d')  # 2行2列的网格，图1
    ax1.plot(pos_data['pos_x'], pos_data['pos_y'], pos_data['pos_z'])
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title(f"{base_filename}_3D Position")

    # 子图2: 2D 轨迹
    ax2 = fig.add_subplot(222)  # 2行2列的网格，图2
    ax2.plot(pos_data['pos_x'], pos_data['pos_y'])
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f"{base_filename}_2D Position")

    # 子图3: 全局加速度随时间
    ax3 = fig.add_subplot(223)  # 2行2列的网格，图3
    ax3.plot(time_data, acce_glob_x, label='Global Acceleration X')
    ax3.plot(time_data, acce_glob_y, label='Global Acceleration Y')
    ax3.plot(time_data, acce_glob_z, label='Global Acceleration Z')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title(f"{base_filename}_Global Acceleration over Time")
        # 设置X轴的范围，聚焦某一段时间的数据（例如 100秒到500秒）
   

    ax3.legend()

    # 子图4: 航向角随时间
    ax4 = fig.add_subplot(224)  # 2行2列的网格，图4
    ax4.plot(time_data, yaw_degrees, label='Yaw (Degrees)', color='orange')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Yaw (Degrees)')
    ax4.set_title(f"{base_filename}_Yaw Degrees over Time")
    ax4.legend()

    # 调整子图之间的布局
    plt.tight_layout()

    # 显示图像
    plt.show(block=True)
