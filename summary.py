import os
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime

def calculate_horizontal_distance(pos1, pos2):
    """计算两个点之间的水平面距离（忽略z轴）"""
    return sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def calculate_3d_distance(pos1, pos2):
    """计算两个点之间的三维空间距离"""
    return sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2 + (pos2[2] - pos1[2]) ** 2)

def calculate_horizontal_speed(distance, time_diff):
    """计算水平速度，不考虑z轴"""
    return distance / time_diff if time_diff > 0 else 0

def calculate_3d_speed(distance, time_diff):
    """计算三维速度，考虑z轴"""
    return distance / time_diff if time_diff > 0 else 0

def format_timestamp(ms):
    """将时间戳（毫秒）转换为 yyyy-mm-dd HH:mm:ss 格式"""
    return datetime.fromtimestamp(ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

def process_csv(file_path):
    """处理单个CSV文件"""
    df = pd.read_csv(file_path)

    # 初始化保存结果的列表
    results = []

    total_horizontal_distance = 0.0
    total_3d_distance = 0.0
    total_time = 0.0

    # 按照1秒时间窗口进行计算
    start_index = 0
    while start_index < len(df):
        # 找到1秒时间窗口内的数据
        start_time = df.loc[start_index, 'timestamp']
        end_time = start_time + 1000  # 1秒 = 1000毫秒
        window_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

        if len(window_df) < 2:
            start_index += len(window_df)
            continue

        # 计算加速度的和与平均值
        acce_x_avg = window_df['acce_glob_x'].mean()
        acce_y_avg = window_df['acce_glob_y'].mean()
        acce_z_avg = window_df['acce_glob_z'].mean()

        # 计算水平位移（忽略z轴）和三维位移
        pos1 = window_df.iloc[0][['pos_x', 'pos_y', 'pos_z']].values
        pos2 = window_df.iloc[-1][['pos_x', 'pos_y', 'pos_z']].values

        horizontal_distance = calculate_horizontal_distance(pos1, pos2)
        total_horizontal_distance += horizontal_distance

        distance_3d = calculate_3d_distance(pos1, pos2)
        total_3d_distance += distance_3d

        # 计算时间差并计算速度
        time_diff = (window_df['timestamp'].iloc[-1] - window_df['timestamp'].iloc[0]) / 1000.0  # 秒
        horizontal_speed = calculate_horizontal_speed(horizontal_distance, time_diff)
        speed_3d = calculate_3d_speed(distance_3d, time_diff)

        total_time += time_diff * 1000  # 转换为毫秒

        # 获取方向角（yaw_degrees），使用时间窗口结束点的yaw
        yaw = window_df.iloc[-1]['yaw_degrees']

        # 保存结果：时间窗口起始时间、加速度的平均值、位置坐标、水平速度、3D速度、水平位移、3D位移、方向角
        results.append({
            '时间': format_timestamp(start_time),  # 时间戳格式化为 %Y-%m-%d %H:%M:%S
            'acce_x_avg': acce_x_avg, 'acce_y_avg': acce_y_avg, 'acce_z_avg': acce_z_avg,
            'pos_x': pos2[0], 'pos_y': pos2[1], 'pos_z': pos2[2],
            '平面速度': horizontal_speed,
            '3d速度': speed_3d,
            '平面位移': horizontal_distance,
            '3d位移': distance_3d,
            '角度': yaw
        })

        # 移动到下一个时间窗口
        start_index += len(window_df)

    avg_speed = total_horizontal_distance / (total_time / 1000.0) if total_time > 0 else 0.0

    # 将结果保存到新的DataFrame
    results_df = pd.DataFrame(results)

    # 生成新的CSV路径
    new_csv_name = os.path.basename(file_path).replace('.csv', '_processed.csv')
    new_csv_path = os.path.join(output_directory, new_csv_name)

    # 将结果保存到新的CSV文件
    results_df.to_csv(new_csv_path, index=False)

    return total_horizontal_distance, total_time, avg_speed

def process_directory(input_dir, output_dir):
    """遍历目录并处理所有CSV文件"""
    results = []

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录及其子目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                total_distance, total_time, avg_speed = process_csv(file_path)
                results.append({
                    'file': file,
                    'total_horizontal_distance': total_distance,
                    'total_time_ms': total_time,
                    'avg_horizontal_speed': avg_speed
                })

    # 保存总体结果到summary.csv
    summary_df = pd.DataFrame(results)
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

if __name__ == '__main__':
    input_directory = os.getcwd() + "/csv"  # 输入目录路径
    output_directory = os.path.join(os.getcwd(), "processed")  # 输出目录路径

    process_directory(input_directory, output_directory)
