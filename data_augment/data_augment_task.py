import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from threading import Event
from PIL import Image, ImageEnhance


# 数据增强任务逻辑
def data_augmentation_task(params, progress_callback, stop_event: Event):
    """模拟数据增强任务"""

    input_path = params.get("input_path")  # 父路径
    output_path = params.get("output_path")  # 输出的父路径
    augmentation_params = params.get("augmentation_params")
    augmentation_count = params.get("augmentation_count")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历所有子目录和文件

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")

        # 找到所有序列
    sequences = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    # sequence_len = len(sequences)
    # sequence_path = os.path.join(input_path, sequences[0])
    # folders = [f for f in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, f))]
    # folder_len = len(folders)
    #
    # folder_path = os.path.join(sequence_path, folders[0])
    # images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    # image_len = len(images)
    #
    # total_steps = sequence_len * folder_len * image_len * augmentation_count

    total_steps = count_images_in_directory(input_path) * augmentation_count


    current_step = 0

    for sequence in sequences:
        sequence_path = os.path.join(input_path, sequence)
        folders = [f for f in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, f))]

        # 随机生成增强参数（一个序列内保持一致）
        seq_params = {
            "brightness": [random.uniform(0.8, 1.5) for _ in range(augmentation_count)] if "brightness" in augmentation_params else None,
            "contrast": [random.uniform(0.7, 1.5) for _ in range(augmentation_count)] if "contrast" in augmentation_params else None,
            "saturation": [random.uniform(0.7, 1.5) for _ in range(augmentation_count)] if "saturation" in augmentation_params else None,
            "hue": [random.uniform(-5, 5) for _ in range(augmentation_count)] if "hue" in augmentation_params else None,
        }

        for folder in folders:

            folder_path = os.path.join(sequence_path, folder)
            output_folder_path = os.path.join(output_path, sequence, folder)
            os.makedirs(output_folder_path, exist_ok=True)

            images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

            for img_file in images:

                if stop_event.is_set():
                    print(f"Task stopped by user. Stopping at file {current_step}/{total_steps}.")

                    return {
                        "status": "paused",
                        "message": "Task stopped.",
                    }
                img_path = os.path.join(folder_path, img_file)
                original_image = Image.open(img_path)

                for augment_index in range(augmentation_count):
                    augmented_image = original_image.copy()

                    # 应用增强
                    if seq_params["brightness"] is not None:
                        enhancer = ImageEnhance.Brightness(augmented_image)
                        augmented_image = enhancer.enhance(seq_params["brightness"][augment_index])
                    if seq_params["contrast"] is not None:
                        enhancer = ImageEnhance.Contrast(augmented_image)
                        augmented_image = enhancer.enhance(seq_params["contrast"][augment_index])
                    if seq_params["saturation"] is not None:
                        enhancer = ImageEnhance.Color(augmented_image)
                        augmented_image = enhancer.enhance(seq_params["saturation"][augment_index])
                    if seq_params["hue"] is not None:
                        # Hue adjustment (Pillow does not natively support this; use workaround if needed)
                        # This is a placeholder for actual hue manipulation

                        augmented_image = np.array(augmented_image)
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2HSV)
                        augmented_image[:, :, 0] = np.clip(augmented_image[:, :, 0] + seq_params["hue"][augment_index], 0, 255)
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_HSV2RGB)
                        augmented_image = Image.fromarray(augmented_image)

                    # 保存增强后的图片
                    output_file_name = f"{os.path.splitext(img_file)[0]}_aug_{augment_index + 1}.jpg"
                    augmented_image.save(os.path.join(output_folder_path, output_file_name))

                    current_step += 1


                progress = int((current_step / total_steps) * 100)
                progress_callback(progress)

    return {
        "status": "success",
        "message": "Task finished.",
    }


def count_images_in_directory(directory, image_extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
    """
    统计指定目录下所有图片的数量（包括子目录）。

    :param directory: 目标目录路径
    :param image_extensions: 图片文件扩展名列表
    :return: 图片数量
    """
    image_count = 0

    # 遍历目录树
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为图片
            if file.lower().endswith(image_extensions):
                image_count += 1

    return image_count


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

    data_augmentation_task(params, None, Event())
