import os
import argparse
import numpy as np
import pandas as pd
import quaternion
import quaternion.quaternion_time_series
from datetime import datetime, timezone
import scipy.interpolate

def interpolate_quaternion_linear(quat_data, input_timestamp, output_timestamp):
    n_input = quat_data.shape[0]
    assert input_timestamp.shape[0] == n_input
    assert quat_data.shape[1] == 4
    n_output = output_timestamp.shape[0]

    quat_inter = np.zeros([n_output, 4])
    ptr1 = 0
    ptr2 = 0
    for i in range(n_output):
        if ptr1 >= n_input - 1 or ptr2 >= n_input:
            raise ValueError("Interpolation error: insufficient input data")
        while input_timestamp[ptr1 + 1] < output_timestamp[i]:
            ptr1 += 1
            if ptr1 == n_input - 1:
                break
        while input_timestamp[ptr2] < output_timestamp[i]:
            ptr2 += 1
            if ptr2 == n_input:
                break
        q1 = quaternion.quaternion(*quat_data[ptr1])
        q2 = quaternion.quaternion(*quat_data[ptr2])
        interpolated_quat = quaternion.quaternion_time_series.slerp(q1, q2, input_timestamp[ptr1],
                                                                     input_timestamp[ptr2],
                                                                     output_timestamp[i])
        quat_inter[i] = quaternion.as_float_array(interpolated_quat)

        # 检查插值结果是否包含 NaN
        if np.isnan(quat_inter[i]).any():
            print(f"Warning: NaN detected in interpolated quaternion at index {i}")
            if i > 0:
                quat_inter[i] = quat_inter[i - 1]  # 使用前一个有效值进行填充
            else:
                quat_inter[i] = np.array([0, 0, 0, 0])  # 使用单位四元数进行填充

    return quat_inter


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.

    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0, fill_value="extrapolate")
    interpolated = func(output_timestamp)
    return interpolated


def process_pose_data(pose_data, new_sampling_rate, skip_front, skip_end):
    # Remove skipped data and swap orientation
    pose_data = pose_data[skip_front:-skip_end]
    pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]

    # Extract position and timestamps
    position = pose_data[:, [1, 3, 2]]
    position[:, 1] *= -1
    timestamps = pose_data[:, 0]

    # Interpolate to new sampling rate
    new_length = int(new_sampling_rate * (timestamps[-1] - timestamps[0]) / 1000)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], new_length)
    
    new_position = interpolate_3dvector_linear(position, timestamps, new_timestamps)

        # 检查输入数据是否包含 NaN
    print("pose_data contains NaN:", np.isnan(pose_data).any())
    print("timestamps contains NaN:", np.isnan(timestamps).any())
    print("new_timestamps contains NaN:", np.isnan(new_timestamps).any())


    new_orientation = interpolate_quaternion_linear(pose_data[:, -4:], timestamps, new_timestamps)

    return new_timestamps, new_position, new_orientation

def process_sensor_data(data, timestamps, new_timestamps):
    return interpolate_3dvector_linear(data[:, 1:], data[:, 0], new_timestamps)

def calculate_heading(sensor_data):

    """
    通过磁力计数据计算航向角度（指南针的角度），范围为0到360度
     'magnet_x': sensor_data['magnet'][:, 0], 'magnet_y': sensor_data['magnet'][:, 1], 'magnet_z': sensor_data['magnet'][:, 2],
    """
    magnet_x = sensor_data['magnet'][:, 0]
    magnet_y =  sensor_data['magnet'][:, 1]

    # 使用 atan2 计算航向角度，结果为弧度
    heading_radians = np.arctan2(magnet_y, magnet_x)

    # 将航向角度从弧度转换为度，并确保角度在 0 到 360 度之间
    heading_degrees = np.degrees(heading_radians)
    heading_degrees = (heading_degrees + 360) % 360

    return heading_degrees

def process_dataset(input_path, output_path, new_sampling_rate=30, skip_front=60, skip_end=60):
    # Load pose data
    pose_file_path = os.path.join(input_path, 'pose.txt')
    
    if not os.path.exists(pose_file_path):
        print(f"Skipping directory {input_path}: pose.txt not found")
        return
    
    # Read the pose data line by line and filter rows
    valid_rows = []
    with open(pose_file_path, 'r') as pose_file:
        for line in pose_file:
            row = line.strip().split()
            if len(row) == 8:
                valid_rows.append([float(val) for val in row])
    
    # Convert valid rows to numpy array
    pose_data = np.array(valid_rows)
    
    # Check if we have any valid data
    if pose_data.shape[0] == 0:
        print(f"Skipping dataset in {input_path}: no valid rows with 8 columns found in pose data")
        return
    
    print(f"Processing dataset in {input_path}")
    print(f"Total rows in pose data: {pose_data.shape[0]}")
    print(f"Rows removed: {sum(1 for line in open(pose_file_path)) - pose_data.shape[0]}")

    # Process pose data
    new_timestamps, new_position, new_orientation = process_pose_data(pose_data, new_sampling_rate, skip_front, skip_end)

    # Process sensor data
    sensors = ['acce', 'gyro', 'linacce', 'gravity', 'magnet']
    sensor_data = {}
    for sensor in sensors:
        sensor_file_path = os.path.join(input_path, f'{sensor}.txt')
        if not os.path.exists(sensor_file_path):
            print(f"Warning: {sensor}.txt not found in {input_path}")
            sensor_data[sensor] = np.zeros((len(new_timestamps), 3))
        else:
            data = np.genfromtxt(sensor_file_path)
            sensor_data[sensor] = process_sensor_data(data, data[:, 0], new_timestamps)

    # Process orientation data
    orientation_file_path = os.path.join(input_path, 'orientation.txt')
    if not os.path.exists(orientation_file_path):
        print(f"Warning: orientation.txt not found in {input_path}")
        output_orientation = np.zeros((len(new_timestamps), 4))
    else:
        orientation_data = np.genfromtxt(orientation_file_path)
        orientation_data[:, [1, 2, 3, 4]] = orientation_data[:, [4, 1, 2, 3]]
        output_orientation = interpolate_quaternion_linear(orientation_data[:, 1:], orientation_data[:, 0], new_timestamps)

    # Convert orientation to quaternion and process global gyro and acce
    init_tango_ori = quaternion.quaternion(*new_orientation[1])
    game_rv = quaternion.from_float_array(output_orientation)

    # Initialize rotor
    init_rotor = init_tango_ori * game_rv[0].conj()
    ori = init_rotor * game_rv  # Final global orientation

    # Global Gyroscope and Accelerometer data
    nz = np.zeros(new_timestamps.shape)
    # gyro_q = quaternion.from_float_array(np.concatenate([nz[:, None], sensor_data['gyro']], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([nz[:, None], sensor_data['acce']], axis=1))

    # gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
    acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

   # 计算航向角（指南针的角度）
    heading_degrees = calculate_heading(sensor_data)

    # Construct DataFrame
    data_dict = {
        'timestamp': new_timestamps,
        'datetime': [datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') for ts in new_timestamps],
        'gyro_x': sensor_data['gyro'][:, 0], 'gyro_y': sensor_data['gyro'][:, 1], 'gyro_z': sensor_data['gyro'][:, 2],
        # 'gyro_glob_x': gyro_glob[:, 0], 'gyro_glob_y': gyro_glob[:, 1], 'gyro_glob_z': gyro_glob[:, 2],
        'acce_x': sensor_data['acce'][:, 0], 'acce_y': sensor_data['acce'][:, 1], 'acce_z': sensor_data['acce'][:, 2],
        'acce_glob_x': acce_glob[:, 0], 'acce_glob_y': acce_glob[:, 1], 'acce_glob_z': acce_glob[:, 2],
        'linacce_x': sensor_data['linacce'][:, 0], 'linacce_y': sensor_data['linacce'][:, 1], 'linacce_z': sensor_data['linacce'][:, 2],
        'grav_x': sensor_data['gravity'][:, 0], 'grav_y': sensor_data['gravity'][:, 1], 'grav_z': sensor_data['gravity'][:, 2],
        'magnet_x': sensor_data['magnet'][:, 0], 'magnet_y': sensor_data['magnet'][:, 1], 'magnet_z': sensor_data['magnet'][:, 2],
        'pos_x': new_position[:, 0], 'pos_y': new_position[:, 1], 'pos_z': new_position[:, 2],
        'ori_w': new_orientation[:, 0], 'ori_x': new_orientation[:, 1], 'ori_y': new_orientation[:, 2], 'ori_z': new_orientation[:, 3],
        # 'ori_glob_w': ori[:, 0].w, 'ori_glob_x': ori[:, 0].x, 'ori_glob_y': ori[:, 0].y, 'ori_glob_z': ori[:, 0].z,
        'rv_w': output_orientation[:, 0], 'rv_x': output_orientation[:, 1], 'rv_y': output_orientation[:, 2], 'rv_z': output_orientation[:, 3],
        'yaw_degrees': heading_degrees 
    }

    df = pd.DataFrame(data_dict)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save to CSV
    output_file = os.path.join(output_path, f'{os.path.basename(input_path)}.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process IMU and pose data")
    parser.add_argument('--input', type=str, default=os.path.join(os.getcwd(), "raw_data"), help='Input directory path')
    parser.add_argument('--output', type=str, default=os.path.join(os.getcwd(), "csv"), help='Output directory path')
    # new_sampling_rate=30, skip_front=60, skip_end=60
    parser.add_argument('--sampling_rate', type=int, default=30, help='New sampling rate')
    parser.add_argument('--skip_front', type=int, default=60, help='Number of samples to skip from the beginning')
    parser.add_argument('--skip_end', type=int, default=60, help='Number of samples to skip from the end')
    args = parser.parse_args()

    # Iterate through all subdirectories in the input directory
    for subdir in os.listdir(args.input):
        subdir_path = os.path.join(args.input, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing directory: {subdir_path}")
            process_dataset(subdir_path, args.output, args.sampling_rate, args.skip_front, args.skip_end)

if __name__ == '__main__':
    main()