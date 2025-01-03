import os

import cv2
import pandas as pd

# 定义评估等级
RESOLUTION_RANK = {
    (224, 224): 1,
    (640, 480): 2,
    (1080, 720): 3
}

# 定义采样频率等级
SAMPLING_FREQUENCY_RANK = {
    2: 1,
    5: 2,
    10: 3,
    20: 4
}


# 定义分辨率评估的连续分布得分
def evaluate_resolution(width, height):
    # 定义分辨率的最高分和最低分
    max_resolution = (1080, 720)  # 最高分辨率
    min_resolution = (224, 224)  # 最低分辨率
    max_score = 100  # 最高分
    min_score = 20  # 最低分

    # 计算分辨率的像素总数
    max_pixels = max_resolution[0] * max_resolution[1]
    min_pixels = min_resolution[0] * min_resolution[1]
    current_pixels = width * height

    # 线性插值计算得分
    if current_pixels >= max_pixels:
        return max_score
    elif current_pixels < min_pixels:
        return 0
    else:
        score = min_score + (current_pixels - min_pixels) / (max_pixels - min_pixels) * (max_score - min_score)
        return round(score, 2)


# 定义采样频率评估的连续分布得分
def evaluate_sampling_frequency(frequency):
    # 定义采样频率的最高分和最低分
    max_frequency = 20  # 最高采样频率
    min_frequency = 2  # 最低采样频率
    max_score = 100  # 最高分
    min_score = 20  # 最低分

    # 线性插值计算得分
    if frequency >= max_frequency:
        return max_score
    elif frequency < min_frequency:
        return 0
    else:
        score = min_score + (frequency - min_frequency) / (max_frequency - min_frequency) * (max_score - min_score)
        return round(score, 2)


# 合规性评估函数
# 合规性评估函数
def compliance_evaluation(dataset_path, video_rate=0.7):
    # 初始化得分字典
    scores = {
        "video": [],
        "csv": []
    }

    # 1. 检查视频文件的分辨率
    video_files = [f for f in os.listdir(dataset_path) if f.endswith(".mp4")]
    for file in video_files:
        file_path = os.path.join(dataset_path, file)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {file_path}")
            scores["video"].append(0)  # 无法打开的视频得分为0
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution_score = evaluate_resolution(width, height)
        scores["video"].append(resolution_score)
        print(f"Resolution of {file}: {width}x{height}, Score: {resolution_score}")

    # 2. 检查CSV文件的采样频率
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)
        try:
            df = pd.read_csv(file_path)
            timestamps = df.iloc[:, 1]  # 假设时间戳在第二列
            time_diff = timestamps.iloc[-1] - timestamps.iloc[0]  # 计算时间差
            num_samples = len(timestamps)
            sampling_frequency = num_samples / time_diff  # 计算采样频率
            frequency_score = evaluate_sampling_frequency(sampling_frequency)
            scores["csv"].append(frequency_score)
            print(f"Sampling frequency of {file}: {sampling_frequency} Hz, Score: {frequency_score}")
        except Exception as e:
            print(f"Error: Failed to process {file}: {e}")
            scores["csv"].append(0)  # 处理失败的CSV得分为0

    # 计算每种类型的平均得分
    video_avg_score = sum(scores["video"]) / len(scores["video"]) if scores["video"] else 0
    print(video_avg_score)
    csv_avg_score = sum(scores["csv"]) / len(scores["csv"]) if scores["csv"] else 0
    print(csv_avg_score)

    # 计算整体得分（按100分计算）
    total_score = video_avg_score * video_rate + csv_avg_score * (1 - video_rate)
    total_score = min(total_score, 100)  # 限制得分不超过100分

    return {
        "video_avg_score": video_avg_score,
        "csv_avg_score": csv_avg_score,
        "total_score": total_score
    }


# 示例调用
dataset_path = r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0"
compliance_score = compliance_evaluation(dataset_path)
print(f"合规性评估得分: {compliance_score} / 20")
