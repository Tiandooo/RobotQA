import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import time
import shutil
from threading import Event

def data_augmentation_task(params, progress_callback, stop_event: Event):
    """模拟数据增强任务"""

    input_path = params.get("input_path")  # 父路径
    output_path = params.get("output_path")  # 输出的父路径
    augmentation_params = params.get("augmentation_params")
    augmentation_count = params.get("augmentation_count")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")

    #--lzh, disable debug mode when publishes.
    debug_file = os.path.join(output_path, "debug.txt")
    debug_mode = True
    #debug_mode = False
    
    # 找到所有序列
    sequences = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    total_steps = count_images_in_directory(input_path) * augmentation_count
    # --lzh, for debug
    if debug_mode:
        with open(debug_file, "w") as debugfile:
            debugfile.write(f"total steps={total_steps}\n")
    current_step = 0

    for sequence in sequences:
        sequence_path = os.path.join(input_path, sequence)
        # --lzh, debug
        if debug_mode:
            with open(debug_file, "a") as debugfile:
                debugfile.write(f"debug: start equence: {sequence_path}\n")
        folders = [f for f in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, f))]

        # 生成增强参数
        seq_params = {
            "brightness": generate_augmentation_values(augmentation_params.get("brightness"), augmentation_count),
            "contrast": generate_augmentation_values(augmentation_params.get("contrast"), augmentation_count),
            "saturation": generate_augmentation_values(augmentation_params.get("saturation"), augmentation_count),
            "hue": generate_augmentation_values(augmentation_params.get("hue"), augmentation_count),
        }

        for folder in folders:
            folder_path = os.path.join(sequence_path, folder)
            output_folder_path = os.path.join(output_path, sequence, folder)    
            # --lzh, debug
            if debug_mode:
                with open(debug_file, "a") as debugfile:
                    debugfile.write(f"debug: entering folder: {folder_path}, save to {output_folder_path}\n")
            if os.path.exists(output_folder_path):
                shutil.rmtree(output_folder_path)
            os.makedirs(output_folder_path, exist_ok=True)

            images = [img for img in os.listdir(folder_path) if img.lower().endswith((".png", ".jpg", ".jpeg"))]

            for img_file in images:
                if stop_event.is_set():
                    print(f"Task stopped by user. Stopping at file {current_step}/{total_steps}.")
                    return {
                        "status": "paused",
                        "message": "Task stopped.",
                    }

                img_path = os.path.join(folder_path, img_file)
                # --lzh, debug
                if debug_mode:
                    with open(debug_file, "a") as debugfile:
                        debugfile.write(f"  processing {img_path}: ")
                original_image = Image.open(img_path)

                for augment_index in range(augmentation_count):
                    # --lzh, debug
                    if debug_mode:
                        with open(debug_file, "a") as debugfile:
                            debugfile.write(f" >> R{augment_index}-")
                    augmented_image = original_image.copy()

                    # 应用增强
                    if seq_params["brightness"] is not None:
                        enhancer = ImageEnhance.Brightness(augmented_image)
                        augmented_image = enhancer.enhance(seq_params["brightness"][augment_index])
                        # --lzh, debug
                        if debug_mode:
                            with open(debug_file, "a") as debugfile:
                                debugfile.write(f"b")
                    if seq_params["contrast"] is not None:
                        enhancer = ImageEnhance.Contrast(augmented_image)
                        augmented_image = enhancer.enhance(seq_params["contrast"][augment_index])
                        # --lzh, debug
                        if debug_mode:
                            with open(debug_file, "a") as debugfile:
                                debugfile.write(f"c")
                    if seq_params["saturation"] is not None:
                        enhancer = ImageEnhance.Color(augmented_image)
                        augmented_image = enhancer.enhance(seq_params["saturation"][augment_index])
                        # --lzh, debug
                        if debug_mode:
                            with open(debug_file, "a") as debugfile:
                                debugfile.write(f"s")
                    if seq_params["hue"] is not None:
                        augmented_image = np.array(augmented_image)
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2HSV)
                        augmented_image[:, :, 0] = np.clip(augmented_image[:, :, 0] + seq_params["hue"][augment_index], 0, 255)
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_HSV2RGB)
                        augmented_image = Image.fromarray(augmented_image)
                        # --lzh, debug
                        if debug_mode:
                            with open(debug_file, "a") as debugfile:
                                debugfile.write(f"h")

                    # 保存增强后的图片
                    output_file_name = f"{os.path.splitext(img_file)[0]}_aug_{augment_index + 1}.jpg"
                    augmented_image.save(os.path.join(output_folder_path, output_file_name))

                    current_step += 1
                    # --lzh, debug
                    if debug_mode:
                        with open(debug_file, "a") as debugfile:
                            debugfile.write(f"-ok.")
                progress = int((current_step / total_steps) * 100)
                progress_callback(progress)
                # --lzh, debug
                if debug_mode:
                    with open(debug_file, "a") as debugfile:
                        debugfile.write(f"{current_step}/{total_steps} done.\n")

    return {
        "status": "success",
        "message": "Task finished.",
    }

def generate_augmentation_values(param_range, count):
    """根据给定的范围生成增强参数值"""
    if param_range is None:
        return None
    if isinstance(param_range, (int, float)):
        # 在单个值附近小幅度波动（例如 ±10%）
        fluctuation = 0.1  # 波动范围
        return [random.uniform(param_range * (1 - fluctuation), param_range * (1 + fluctuation)) for _ in range(count)]
    if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
        return [random.uniform(param_range[0], param_range[1]) for _ in range(count)]
    raise ValueError("Invalid parameter range format. Expected None, a single value, or a list/tuple of two values.")

def count_images_in_directory(directory):
    """计算目录中的图片数量"""
    count = 0
    for root, dirs, files in os.walk(directory):
        for allfile in files:
            if allfile.lower().endswith((".png", ".jpg", ".jpeg")):
                count += 1
    return count


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
        for allfile in files:
            # 检查文件扩展名是否为图片
            if allfile.lower().endswith(image_extensions):
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
