import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import ImageEnhance, Image
import random
from tqdm import tqdm

from utils.load_images import load_images

class ImageAugmenter:
    def __init__(self, image_directory: str, save_directory: str, adjust_before: bool = True):
        # 初始化增强器，设置图像目录和增强参数
        self.image_directory = image_directory
        self.save_directory = save_directory

        # 增强前是否调整亮度（对整条episode的亮度平滑处理）
        self.adjust_before = adjust_before

        # 加载图像，并获取最大episode数
        self.image_grid, self.max_episode = load_images(image_directory)

    def color_jitter(self, image: Image.Image, brightness=1.0, contrast=1.0, saturation=1.0, hue=0) -> Image.Image:
        """
        调整图像的亮度、对比度、饱和度和色调。
        """
        # 调整亮度
        image = ImageEnhance.Brightness(image).enhance(brightness)
        # 调整对比度
        image = ImageEnhance.Contrast(image).enhance(contrast)
        # 调整饱和度
        image = ImageEnhance.Color(image).enhance(saturation)

        # 调整色调
        if hue != 0:
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image[:, :, 0] = np.clip(image[:, :, 0] + hue, 0, 255)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            image = Image.fromarray(image)

        return image

    def random_noise(self, image: Image, mean=0, std=0.2) -> Image:
        """
        给图像添加随机噪声。
        """
        image_array = np.array(image)
        # 创建高斯噪声
        noise = np.random.normal(mean, std, image_array.shape).astype(np.uint8)
        # 将噪声加到图像上，确保不超出[0, 255]范围
        noisy_image = np.clip(image_array + noise, 0, 255)
        return Image.fromarray(noisy_image)

    def generate_random_augmentation(self):
        """
        随机生成增强参数组合。
        """
        brightness = random.uniform(0.8, 1.5)  # 随机亮度调整
        contrast = random.uniform(0.7, 1.5)    # 随机对比度调整
        saturation = random.uniform(0.7, 1.5)  # 随机饱和度调整
        hue = random.randint(-5, 5)            # 随机色调调整
        add_noise = random.choice([True, False])  # 随机决定是否添加噪声

        # 返回一个增强操作链
        aug_chain = [lambda img: self.color_jitter(img, brightness, contrast, saturation, hue)]
        if add_noise:
            aug_chain.append(self.random_noise)

        return aug_chain

    def apply_augmentations(self, num_augments: int = 3, target_brightness: int = 128):
        """
        应用随机增强操作并保存增强后的图像。
        """
        if not os.path.isdir(self.save_directory):
            os.mkdir(self.save_directory)

        current_episode = self.max_episode + 1
        # 遍历每一集（episode）的图像
        for episode_index, episode_images in tqdm(enumerate(self.image_grid), desc="Applying Augmentations"):
            if self.adjust_before:

                episode_images = self.equalize_epoch_brightness(episode_images, target_brightness=target_brightness)
            for aug_idx in range(num_augments):
                # 生成一个随机增强链
                aug_chain = self.generate_random_augmentation()
                for step, img in enumerate(episode_images):
                    aug_img = img
                    # 对每张图像应用所有增强操作
                    for aug in aug_chain:
                        aug_img = aug(aug_img)

                    # 保存增强后的图像
                    new_filename = f'sri_kitchen_T0319_main_{episode_index}_step_{step}_aug{aug_idx}.png'
                    new_image_path = os.path.join(self.save_directory, new_filename)
                    aug_img.save(new_image_path)
                    aug_img.close()
    """
    调整亮度
    """
    def calculate_brightness(self, image: Image.Image) -> float:
        """
        计算图像的亮度值（通过将图像转换为灰度图像）。
        """
        gray_image = image.convert("L")
        brightness = np.mean(np.array(gray_image))
        return brightness

    def adjust_brightness(self, image: Image.Image, target_brightness: float) -> Image.Image:
        """
        调整图像的亮度，使其接近目标亮度。
        """
        current_brightness = self.calculate_brightness(image)
        if current_brightness == 0:
            return image
        brightness_factor = target_brightness / current_brightness
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_factor)

    def equalize_epoch_brightness(self, images: List[Image.Image], target_brightness: float = 128.0) -> List[Image.Image]:
        """
        将一个epoch的图像亮度均衡到目标亮度。
        """
        adjusted_images = []
        for image in images:
            adjusted_image = self.adjust_brightness(image, target_brightness)
            adjusted_images.append(adjusted_image)
        return adjusted_images


if __name__ == "__main__":

    image_directory = '../data/images'
    save_directory = '../data/images_right_aug2'
    augmenter = ImageAugmenter(image_directory, save_directory, adjust_before=True)
    augmenter.apply_augmentations(num_augments=1, target_brightness=128)
