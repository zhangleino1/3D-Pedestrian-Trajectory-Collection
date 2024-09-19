import os
import pandas as pd

def process_csv_file(file_path, initial_position):
    df = pd.read_csv(file_path)

    # 提取pos_x, pos_y, pos_z列
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    pos_z = df['pos_z'].values

    # 初始化真实位置列表
    real_pos_x = [initial_position[0]]
    real_pos_y = [initial_position[1]]
    real_pos_z = [initial_position[2]]

    # 遍历数据行，从第二行开始
    for i in range(1, len(df)):
        # 计算相对位移
        delta_x = pos_x[i] - pos_x[i-1]
        delta_y = pos_y[i] - pos_y[i-1]
        delta_z = pos_z[i] - pos_z[i-1]

        # 计算当前的真实位置
        real_x = real_pos_x[-1] + delta_x
        real_y = real_pos_y[-1] + delta_y
        real_z = real_pos_z[-1] + delta_z

        real_pos_x.append(real_x)
        real_pos_y.append(real_y)
        real_pos_z.append(real_z)

    # 更新DataFrame中的位置列
    df['pos_x'] = real_pos_x
    df['pos_y'] = real_pos_y
    df['pos_z'] = real_pos_z

    # 保存更新后的CSV文件，可以选择覆盖原文件或另存为新文件
    df.to_csv(file_path, index=False)
    print(f'已处理文件：{file_path}')

def process_directory(directory_path, initial_position):
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                process_csv_file(file_path, initial_position)

# 示例调用
if __name__ == '__main__':
    # 设置初始位置（室内真实位置），例如 [0.0, 0.0, 0.0]
    initial_position = [0.0, 0.0, 0.0]  # 请根据实际情况修改

    # 设置要处理的CSV文件所在的目录
    directory_path = os.getcwd() + "/csv"  # 请修改为您的实际目录路径

    process_directory(directory_path, initial_position)
