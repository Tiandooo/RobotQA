import json
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from threading import Event
from PIL import Image, ImageEnhance

from data_qa.vqa import get_video_quality


# 数据增强任务逻辑
def data_qa_task(params, progress_callback, stop_event: Event):
    """模拟数据增强任务"""

    input_path = params.get("input_path")  # 父路径
    output_path = params.get("output_path")  # 输出的父路径
    device = params.get("device")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历所有子目录和文件

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")

        # 找到所有序列
    sequences = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    sequence_len = len(sequences)
    sequence_path = os.path.join(input_path, sequences[0])
    # folders = [f for f in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, f))]
    # folder_len = len(folders)

    # folder_path = os.path.join(sequence_path, folders[0])
    # images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    videos = [video for video in os.listdir(sequence_path) if video.endswith('.mp4')]
    # image_len = len(images)
    video_len = len(videos)
    print(sequence_len, video_len)

    total_steps = sequence_len * video_len

    current_step = 0

    for sequence in sequences:
        sequence_path = os.path.join(input_path, sequence)

        output_folder_path = os.path.join(output_path, sequence)
        os.makedirs(output_folder_path, exist_ok=True)
        # images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        videos = [video for video in os.listdir(sequence_path) if video.endswith('.mp4')]
        scores = {}
        for video in videos:
            if stop_event.is_set():
                print(f"Task stopped by user. Stopping at file {current_step}/{total_steps}.")
                return {
                    "status": "paused",
                    "message": "Task stopped.",
                }
            video_path = os.path.join(sequence_path, video)
            score = get_video_quality(video_path, device=device)
            scores[video] = score
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_callback(progress)
        with open(os.path.join(output_folder_path, "cores.json"), "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)


    return {
        "status": "success",
        "message": "Task finished.",
    }


if __name__ == "__main__":
    params = {
        "input_path": "../data/kitchen_40demo/raw_data",
        "output_path": "../data/kitchen_40demo/aug",
        "augmentation_params": {
            "brightness": True,
            "contrast": True,
            "saturation": False,
            "hue": True
        },
        "augmentation_count": 3
    }

    # data_augmentation_task(params, None, Event())
