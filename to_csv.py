import os
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pandas as pd
import numpy as np

def parse_pose_file(pose_file_path):
    """
    解析pose.txt文件，将数据转换为DataFrame格式
    """
    valid_rows = []
    
    # 打开文件并逐行读取
    with open(pose_file_path, 'r') as file:
        for line in file:
            # 分割每一行数据为列表
            columns = line.strip().split()
            
            # 如果行数据包含8列，保留它
            if len(columns) == 8:
                valid_rows.append([float(i) for i in columns])
    
    # 如果没有有效的数据行，返回空的DataFrame
    if not valid_rows:
        return pd.DataFrame()
    
    # 转换为numpy数组
    data = np.array(valid_rows)
    
    # 交换四元数的顺序，从 [x, y, z, w] 到 [w, x, y, z]
    data[:, [-4, -3, -2, -1]] = data[:, [-1, -4, -3, -2]]
    
    # 处理位置数据
    position = np.zeros([len(data), 3])
    position[:, 0] = data[:, 1]  # pos_x
    position[:, 1] = -data[:, 3]  # pos_y，Y轴需要取反
    position[:, 2] = data[:, 2]  # pos_z

    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': data[:, 0],
        'pos_x': position[:, 0],
        'pos_y': position[:, 1],
        'pos_z': position[:, 2],
        'quat_w': data[:, 4],
        'quat_x': data[:, 5],
        'quat_y': data[:, 6],
        'quat_z': data[:, 7]
    })
    
    return df


def calculate_frequency(df):
    """
    计算每秒采集的频率，并将时间戳转换为年月日时分秒的格式
    """
    print( df['timestamp'])
    # 将时间戳从毫秒转换为秒，并转换为日期时间格式
    df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    
    # 统计每秒的频率
    frequency = df.groupby('datetime').size()
    
    return frequency
def process_directory(input_dir, output_dir):
    """
    遍历目录，找到所有pose.txt文件，并将其转换为CSV格式，保存到输出目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录及其子目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == 'pose.txt':
                # 完整的pose.txt文件路径
                pose_file_path = os.path.join(root, file)
                
                # 解析pose.txt文件
                df = parse_pose_file(pose_file_path)
                
                # 计算采样频率
                frequency = calculate_frequency(df)
                
                # 打印每秒的采样频率
                print(f"Frequency for {os.path.basename(root)}:")
                print(frequency)
                print()
                
                # 获取当前文件夹名称
                folder_name = os.path.basename(root)
                
                # 确定输出CSV文件路径
                output_csv_path = os.path.join(output_dir, f'{folder_name}.csv')
                
                # 将DataFrame保存为CSV文件
                df.to_csv(output_csv_path, index=False)
                print(f"Saved CSV to {output_csv_path}")

if __name__ == '__main__':
    input_directory = os.getcwd()  # 输入目录路径
    output_directory = os.path.join(os.getcwd(), "csv")  # 输出目录路径
    
    process_directory(input_directory, output_directory)
    
    
