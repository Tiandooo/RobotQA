import cv2
import pandas as pd
import os


# 一致性评估函数
def evaluate_consistency(video_file, csv_file):
    # 初始化得分
    consistency_score = 100.0

    # 1. 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
        frame_indices = df['frame_idx'].tolist()  # 获取CSV中的帧索引
        timestamps = df['timestamp'].tolist()    # 获取CSV中的时间戳
    except Exception as e:
        print(f"Error: Failed to read CSV file {csv_file}: {e}")
        return 0.0  # 如果无法读取CSV文件，一致性得分为0

    # 2. 读取视频文件
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_file}")
        return 0.0  # 如果无法打开视频文件，一致性得分为0

    # 获取视频的总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 3. 检查视频帧与CSV帧索引的一致性
    if total_frames != len(frame_indices):
        print(f"Warning: Video frame count ({total_frames}) does not match CSV frame count ({len(frame_indices)})")
        consistency_score -= 50  # 严重不一致，扣50分

    # 4. 检查视频时间戳与CSV时间戳的一致性
    video_timestamps = [i / fps for i in range(total_frames)]  # 计算视频的理论时间戳
    csv_timestamps = timestamps[:total_frames]  # 只取与视频帧数相同的时间戳

    # 计算时间戳的差异
    max_time_diff = 0.1  # 允许的最大时间戳差异（秒）
    for video_ts, csv_ts in zip(video_timestamps, csv_timestamps):
        time_diff = abs(video_ts - csv_ts)
        if time_diff > max_time_diff:
            print(f"Warning: Timestamp mismatch (Video: {video_ts}, CSV: {csv_ts})")
            consistency_score -= 10  # 时间戳不一致，扣10分

    # 5. 返回一致性得分
    consistency_score = max(consistency_score, 0.0)  # 得分不低于0
    return consistency_score

# 示例调用
dataset_path = r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0"
video_file = os.path.join(dataset_path, "wrist_image_1.mp4")
csv_file = os.path.join(dataset_path, "cartesian_joint.csv")

consistency_score = evaluate_consistency(video_file, csv_file)
print(f"一致性评估得分: {consistency_score} / 100")
