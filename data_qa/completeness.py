import os
import cv2
import pandas as pd


# 完整性评估函数
def completeness_evaluation(dataset_path):
    completeness_score = 20  # 完整性评估满分

    # 1. 检查视频和图像是否存在
    video_files = [
        f"{dataset_path}/exterior_image_1.mp4",
        f"{dataset_path}/wrist_image_1.mp4"
    ]
    image_files = [
        f"{dataset_path}/exterior_image_1",
        f"{dataset_path}/wrist_image_1"
    ]
    for file in video_files + image_files:
        if not os.path.exists(file):
            if file.endswith(".mp4"):
                completeness_score -= 10  # 缺失视频严重扣分
            else:
                completeness_score -= 2  # 缺失图像扣分
            print(f"Warning: File {file} is missing.")

    # 2. 检查是否跳帧
    for file in video_files:
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {file}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        duration = frame_count / fps
        expected_frames = duration * fps
        if abs(frame_count - expected_frames) > 1:
            completeness_score -= 2
            print(f"Warning: Video {file} has skipped frames.")

    # 3. 检查其他模态的数据是否存在
    data_files = [
        "cartesian_position.csv",
        "cartesian_joint.csv",
        "joint_torque.csv",
        "gripper_position.csv"
    ]
    for file in data_files:
        if not os.path.exists(f"{dataset_path}/{file}"):
            completeness_score -= 2
            print(f"Warning: File {file} is missing.")

    # 计算最终得分
    completeness_score = max(completeness_score, 0)
    return completeness_score

# 示例调用
completeness_score = completeness_evaluation(r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0")
print(f"完整性评估得分: {completeness_score} / 20")
