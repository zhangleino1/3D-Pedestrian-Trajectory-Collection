import os
import pandas as pd
import numpy as np
from math import sqrt, atan2, degrees

def calculate_horizontal_distance(pos1, pos2):
    """计算两个点之间的水平面距离（忽略z轴）"""
    return sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def calculate_horizontal_angle(pos1, pos2):
    """计算两个点之间的水平面角度变化（单位：度）"""
    delta_x = pos2[0] - pos1[0]
    delta_y = pos2[1] - pos1[1]
    return degrees(atan2(delta_y, delta_x))

def process_csv(file_path):
    """处理单个CSV文件"""
    df = pd.read_csv(file_path)

    # 初始化保存结果的列表
    results = []

    total_horizontal_distance = 0.0
    total_time = 0.0
    last_angle = None

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

        # 计算窗口内的总距离
        distances = []
        for i in range(1, len(window_df)):
            pos1 = window_df.iloc[i-1][['pos_x', 'pos_y']].values
            pos2 = window_df.iloc[i][['pos_x', 'pos_y']].values
            distances.append(calculate_horizontal_distance(pos1, pos2))

        horizontal_distance = sum(distances)
        total_horizontal_distance += horizontal_distance

        # 计算平均速度
        time_diff = (window_df['timestamp'].iloc[-1] - window_df['timestamp'].iloc[0]) / 1000.0  # 秒
        horizontal_speed = horizontal_distance / time_diff if time_diff > 0 else 0.0

        # 计算水平角度变化
        current_angle = calculate_horizontal_angle(window_df.iloc[0][['pos_x', 'pos_y']].values, 
                                                   window_df.iloc[-1][['pos_x', 'pos_y']].values)
        if last_angle is not None:
            angle_change = abs(current_angle - last_angle)
        else:
            angle_change = 0.0
        last_angle = current_angle

        # 保存结果：窗口起始时间，起始位置，水平速度，水平距离，角度变化
        results.append({
            'timestamp': start_time,
            'start_pos_x': window_df.iloc[0]['pos_x'],
            'start_pos_y': window_df.iloc[0]['pos_y'],
            'horizontal_speed': horizontal_speed,
            'horizontal_distance': horizontal_distance,
            'horizontal_angle_change': angle_change
        })

        # 移动到下一个时间窗口
        start_index += len(window_df)

    avg_speed = total_horizontal_distance / total_time if total_time > 0 else 0.0

    # 将结果保存到新的DataFrame
    results_df = pd.DataFrame(results)

    # 生成新的CSV路径
    new_csv_name = os.path.basename(file_path).replace('.csv', '_processed.csv')
    new_csv_path = os.path.join(output_directory, new_csv_name)

    # 将结果保存到新的CSV文件
    results_df.to_csv(new_csv_path, index=False)

    return total_horizontal_distance, avg_speed

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
                total_distance, avg_speed = process_csv(file_path)
                results.append({
                    'file': file,
                    'total_horizontal_distance': total_distance,
                    'avg_horizontal_speed': avg_speed
                })

    # 保存总体结果到summary.csv
    summary_df = pd.DataFrame(results)
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

if __name__ == '__main__':
    input_directory = os.getcwd()+"/csv"  # 输入目录路径
    output_directory = os.path.join(os.getcwd(), "processed")  # 输出目录路径

    process_directory(input_directory, output_directory)
