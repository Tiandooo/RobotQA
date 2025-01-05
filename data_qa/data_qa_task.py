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


def data_qa_task(params, progress_callback, stop_event: Event):
    """
    数据质量评估任务。
    """
    input_path = params.get("input_path")  # 父路径
    output_path = params.get("output_path")  # 输出的父路径
    device = params.get("device")
    qa_params = params.get("qa_params")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")

    # 找到所有序列
    sequences = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    sequence_len = len(sequences)
    sequence_path = os.path.join(input_path, sequences[0])
    videos = [video for video in os.listdir(sequence_path) if video.endswith('.mp4')]
    video_len = len(videos)
    print(sequence_len, video_len)

    total_steps = sequence_len * video_len
    current_step = 0

    for sequence in sequences:
        sequence_path = os.path.join(input_path, sequence)
        output_folder_path = os.path.join(output_path, sequence)
        os.makedirs(output_folder_path, exist_ok=True)

        videos = [video for video in os.listdir(sequence_path) if video.endswith('.mp4')]
        # 设置默认得分
        scores = {

        }
        if qa_params.get("completeness_score"):
            scores["completeness_score"] = 100
        # 准确性评分逻辑
        if qa_params.get("accuracy_score"):
            scores["accuracy_score"] = 100

            accuracy_score = 0
            video_count = 0
            for video in videos:
                if stop_event.is_set():
                    print(f"Task stopped by user. Stopping at file {current_step}/{total_steps}.")
                    return {
                        "status": "paused",
                        "message": "Task stopped.",
                    }

                video_path = os.path.join(sequence_path, video)

                # 获取视频质量得分
                score = get_video_quality(video_path, device=device) * 100

                # 更新accuracy得分
                accuracy_score = (accuracy_score * video_count + score) / (video_count + 1)
                video_count += 1

                # 保存得分

                current_step += 1
                progress = int((current_step / total_steps) * 100)
                progress_callback(progress)

            scores["accuracy_score"] = accuracy_score
        if qa_params.get("consistency_score"):
            scores["consistency_score"] = 100

        if qa_params.get("accessable_score"):
            scores["accessable_score"] = 100


        # TODO: 加上其它的score

        # 将得分保存到 JSON 文件
        with open(os.path.join(output_folder_path, "scores.json"), "w", encoding="utf-8") as f:
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
