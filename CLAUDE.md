# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **3D Pedestrian Trajectory Collection** project designed to collect, process, and analyze pedestrian trajectory data in three-dimensional space. The project focuses on detailed recording of pedestrian movements with 1-second granularity, analyzing speed, angle (yaw), and distance traveled.

The system uses IMU sensor data (accelerometer, gyroscope, magnetometer, linear acceleration, gravity) combined with pose tracking to reconstruct accurate 3D trajectories.

## Data Pipeline Architecture

The project follows a three-stage data processing pipeline:

1. **Raw Data Collection** (`raw_data/` directory):
   - Each subdirectory (named with timestamp like `20240829_225513`) contains sensor recordings
   - Key sensor files: `pose.txt`, `acce.txt`, `gyro.txt`, `linacce.txt`, `gravity.txt`, `magnet.txt`, `orientation.txt`
   - Pose data format: 8-column format (timestamp, x, y, z, orientation quaternion)

2. **CSV Conversion** (`csv/` directory):
   - Processed via `to_csv.py`
   - Interpolates sensor data to uniform sampling rate (default 180Hz)
   - Synchronizes all sensors to common timestamp
   - Computes global coordinate transformations using quaternion math
   - Calculates yaw degrees from orientation quaternions

3. **Statistical Analysis** (`processed/` directory):
   - Processed via `summary.py`
   - Aggregates data into 1-second time windows
   - Computes speed, distance, and acceleration statistics
   - Generates per-trajectory summaries

## Key Commands

### Data Processing

```bash
# Convert raw sensor data to CSV (default settings: 180Hz sampling, skip first/last 60 samples)
python to_csv.py

# With custom parameters
python to_csv.py --input ./raw_data --output ./csv --sampling_rate 180 --skip_front 60 --skip_end 60

# Process CSV data to generate statistics and summaries
python summary.py

# Convert relative positions to absolute positions based on initial location
python gt_process.py
```

### Data Visualization

```bash
# Generate 3D/2D trajectory plots, acceleration, and yaw angle graphs
python trace_view.py
```

This creates visualizations with 4 subplots:
- 3D trajectory plot
- 2D trajectory (X-Y plane)
- Global acceleration over time
- Yaw degrees over time

## Important Technical Details

### Quaternion Processing
The codebase heavily uses quaternions for orientation handling:
- Uses `numpy-quaternion` library for quaternion operations
- Performs SLERP (Spherical Linear Interpolation) for orientation interpolation
- Converts between local and global coordinate frames using quaternion conjugates
- Computes yaw angle from quaternion orientation data

### Coordinate System Transformations
- Tango pose coordinates are transformed: swaps axes and inverts Y-axis
- Orientation quaternion elements are reordered: `[w,x,y,z]` → `[x,y,z,w]` → `[w,x,y,z]`
- Global accelerometer data is computed by rotating local measurements using orientation quaternions

### Data Interpolation
- All sensors are interpolated to a common sampling rate (default 180Hz)
- Position uses linear interpolation (`scipy.interpolate.interp1d`)
- Orientation uses SLERP (Spherical Linear Interpolation) to preserve quaternion properties
- NaN values in quaternion interpolation are handled by forward-filling or using identity quaternion

### Time Window Processing (summary.py)
- Uses 1-second time windows for statistical aggregation
- Computes both 2D (horizontal plane) and 3D metrics:
  - Distance: accumulated within window
  - Speed: averaged across all sub-intervals in window
  - Acceleration: averaged across entire window
- Tracks cumulative distance and time across entire trajectory

## Directory Structure

```
3D-Pedestrian-Trajectory-Collection/
├── raw_data/              # Raw sensor data (subdirectories named by timestamp)
│   └── YYYYMMDD_HHMMSS/  # Each collection session
│       ├── pose.txt       # Position and orientation (8 columns)
│       ├── acce.txt       # Accelerometer
│       ├── gyro.txt       # Gyroscope
│       ├── linacce.txt    # Linear acceleration
│       ├── gravity.txt    # Gravity vector
│       ├── magnet.txt     # Magnetometer
│       └── orientation.txt # Device orientation
├── csv/                   # Processed CSV files (one per session)
├── processed/             # Statistical summaries
│   └── summary.csv        # Overall trajectory statistics
├── image/                 # Generated visualizations
├── to_csv.py             # Raw data → CSV converter
├── summary.py            # CSV → Statistics processor
├── gt_process.py         # Converts relative to absolute positions
└── trace_view.py         # Visualization generator
```

## Common Issues

### NaN Values in Quaternion Interpolation
If quaternion interpolation produces NaN values (to_csv.py:38-43), the code handles it by:
1. Forward-filling with the previous valid quaternion
2. Using identity quaternion `[0,0,0,0]` if no previous valid value exists

### Pose Data Format Validation
`to_csv.py` filters out malformed rows that don't have exactly 8 columns (lines 133-145). Check raw `pose.txt` files if datasets are skipped.

### Deprecated Pandas Methods
`trace_view.py:34` uses deprecated `fillna(method='ffill')`. Modern equivalent: `fillna(method='ffill')` → `ffill()`

## Output Data Schema

### CSV Files (from to_csv.py)
Key columns:
- `timestamp`: Unix timestamp in milliseconds
- `datetime`: Human-readable timestamp
- `pos_x`, `pos_y`, `pos_z`: 3D position in meters
- `acce_glob_x/y/z`: Global frame acceleration (m/s²)
- `gyro_x/y/z`: Angular velocity (rad/s)
- `yaw_degrees`: Heading angle (0-360°)
- `ori_w/x/y/z`: Orientation quaternion
- Sensor-specific columns for magnetometer, gravity, linear acceleration

### Processed Files (from summary.py)
Key columns:
- `时间`: Formatted timestamp (YYYY-MM-DD HH:MM:SS)
- `acce_x/y/z_avg`: Average acceleration in 1-second window
- `pos_x/y/z`: Position at end of window
- `平均平面速度`: Average horizontal speed (m/s)
- `平均3d速度`: Average 3D speed (m/s)
- `总平面位移`: Total horizontal distance in window (m)
- `总3d位移`: Total 3D distance in window (m)
- `角度`: Yaw angle at end of window (degrees)
