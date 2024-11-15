import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import ImageEnhance, Image
import random
from tqdm import tqdm

def load_images(image_directory: str) -> Tuple[List[List[Image.Image]], int]:
    """
    加载图像并按行、列组织。每个episode为一行，step为列。
    """
    image_filenames = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')])
    image_grid = []
    current_row = []
    current_episode = 0

    for filename in image_filenames:
        # 分割文件名提取 episode 和 step 信息
        parts = filename.replace('.png', '').split('_')
        if "aug" in parts[-1]:  # 跳过已经增强过的图片
            continue
        episode = int(parts[-3])
        step = int(parts[-1])

        # 打开图片
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        # 如果遇到新的 episode，开始新行
        if episode != current_episode:
            if current_row:
                image_grid.append(current_row)
            current_row = []
            current_episode = episode

        current_row.append(image)

    if current_row:
        image_grid.append(current_row)

    return image_grid, current_episode