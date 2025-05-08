import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from threading import Event

import os
import shutil
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from threading import Event

import os
import shutil
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import CubicSpline
from threading import Event
from tqdm import tqdm
from flask import jsonify


def generate_new_timestamps(original_timestamps, interpolation_factor):
    """生成插值后的新时间序列"""
    new_timestamps = []
    for i in range(len(original_timestamps) - 1):
        start = original_timestamps[i]
        end = original_timestamps[i + 1]
        num_points = interpolation_factor + 1
        segment = np.linspace(start, end, num=num_points + 1)[:-1]
        new_timestamps.extend(segment)
    new_timestamps.append(original_timestamps[-1])
    return np.array(new_timestamps)


def process_csv(input_path, output_path, filename, t_original, t_new, progress_callback):
    """处理单个CSV文件"""
    df = pd.read_csv(os.path.join(input_path, filename))

    # 提取数值列
    cols = [c for c in df.columns if c not in ['frame_idx', 'timestamp']]

    # 创建插值函数
    interpolators = {col: CubicSpline(t_original, df[col]) for col in cols}

    # 生成新数据
    new_data = {col: interpolators[col](t_new) for col in cols}
    new_df = pd.DataFrame({
        'frame_idx': np.arange(len(t_new)),
        'timestamp': t_new,
        **new_data
    })

    # 保持原始列顺序
    new_df = new_df[df.columns]
    new_df.to_csv(os.path.join(output_path, filename), index=False)
    progress_callback(10)  # 每个CSV文件占10%进度


def interpolate_frames(prev_frame, next_frame, alpha):
    """使用光流法进行帧插值"""
    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算双向光流
    flow_forward = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_backward = cv2.calcOpticalFlowFarneback(
        next_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 创建网格坐标
    h, w = prev_gray.shape
    y, x = np.mgrid[0:h:1, 0:w:1].astype(np.float32)

    # 前向变形
    forward_coords = np.dstack((x + flow_forward[..., 0], y + flow_forward[..., 1]))
    forward_warp = cv2.remap(prev_frame, forward_coords, None, cv2.INTER_LINEAR)

    # 反向变形
    backward_coords = np.dstack((x + flow_backward[..., 0], y + flow_backward[..., 1]))
    backward_warp = cv2.remap(next_frame, backward_coords, None, cv2.INTER_LINEAR)

    # 线性混合
    return cv2.addWeighted(forward_warp, 1 - alpha, backward_warp, alpha, 0)


def process_images(input_dir, output_dir, t_original, t_new, frame_indices, progress_callback):
    """处理图像目录"""
    os.makedirs(output_dir, exist_ok=True)
    original_images = {
        int(f.split('.')[0]): os.path.join(input_dir, f)
        for f in os.listdir(input_dir) if f.endswith('.jpg')
    }

    for new_idx, t in tqdm(enumerate(t_new), desc="Processing images"):
        pos = np.searchsorted(t_original, t, side='right') - 1
        pos = max(0, min(pos, len(t_original) - 2))

        prev_time = t_original[pos]
        next_time = t_original[pos + 1]

        if t == prev_time:
            src_idx = frame_indices[pos]
            src_path = original_images.get(src_idx)
            if src_path:
                shutil.copy(src_path, os.path.join(output_dir, f"{new_idx:06d}.jpg"))
            continue
        elif t == next_time:
            src_idx = frame_indices[pos + 1]
            src_path = original_images.get(src_idx)
            if src_path:
                shutil.copy(src_path, os.path.join(output_dir, f"{new_idx:06d}.jpg"))
            continue

        # 计算插值权重
        alpha = (t - prev_time) / (next_time - prev_time)

        # 读取相邻帧
        prev_idx = frame_indices[pos]
        next_idx = frame_indices[pos + 1]
        prev_frame = cv2.imread(original_images[prev_idx])
        next_frame = cv2.imread(original_images[next_idx])

        if prev_frame is None or next_frame is None:
            continue

        # 生成中间帧
        interpolated = interpolate_frames(prev_frame, next_frame, alpha)
        cv2.imwrite(os.path.join(output_dir, f"{new_idx:06d}.jpg"), interpolated)

    progress_callback(20)  # 每个图像目录占20%进度


def multi_modal_data_augmentation_task(params, progress_callback, stop_event: Event):
    try:
        input_path = params["input_path"]
        output_path = params["output_path"]
        augmentation_params = params.get("augmentation_params", {})

        interpolation_factor = augmentation_params.get("interpolation_factor", 1)
        os.makedirs(output_path, exist_ok=True)

        # 步骤计数器
        steps_completed = 0
        total_steps = 4 + 2 * 2  # CSV文件 + 图像目录 + 视频处理

        def update_progress(increment=0):
            nonlocal steps_completed
            steps_completed += increment
            progress = int((steps_completed / total_steps) * 100)
            progress_callback(min(progress, 100))

        # 读取基准时间戳
        base_csv = os.path.join(input_path, "cartesian_joint.csv")
        df_base = pd.read_csv(base_csv)
        t_original = df_base["timestamp"].values
        frame_indices = df_base["frame_idx"].values

        # 生成新时间序列
        t_new = generate_new_timestamps(t_original, interpolation_factor)
        update_progress(10)

        # 处理CSV文件
        csv_files = [
            "cartesian_joint.csv",
            "cartesian_position.csv",
            "joint_torque.csv",
            "gripper_position.csv"
        ]

        for csv_file in csv_files:
            if stop_event.is_set():
                return {"status": "stopped", "message": "Process interrupted"}
            process_csv(input_path, output_path, csv_file, t_original, t_new, lambda: update_progress(10))

        # 处理图像目录
        image_dirs = ["exterior_image_1", "wrist_image_1"]
        for img_dir in image_dirs:
            if stop_event.is_set():
                return {"status": "stopped", "message": "Process interrupted"}
            input_dir = os.path.join(input_path, img_dir)
            output_dir = os.path.join(output_path, img_dir)
            process_images(input_dir, output_dir, t_original, t_new, frame_indices, lambda: update_progress(20))

        # 处理视频文件
        video_files = ["exterior_image_1.mp4", "wrist_image_1.mp4"]
        for video_file in video_files:
            if stop_event.is_set():
                return {"status": "stopped", "message": "Process interrupted"}
            img_dir = video_file.split(".")[0]
            image_folder = os.path.join(output_path, img_dir)
            video_path = os.path.join(output_path, video_file)

            images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
            images.sort()

            if not images:
                continue

            # 获取视频参数
            first_image = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, _ = first_image.shape
            fps = 5 * (interpolation_factor + 1)

            # 使用h264编码
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for image in tqdm(images, desc=f"Writing {video_file}"):
                img_path = os.path.join(image_folder, image)
                frame = cv2.imread(img_path)
                video.write(frame)

            video.release()
            update_progress(10)

        return {"status": "completed", "message": "Data augmentation finished successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e)}