import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
class ImageSafetyChecker:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 多标签分类模型（暴力、色情、仇恨内容等）
        self.nsfw_detector = pipeline(
            "image-classification",
            model="checkpoints/nsfw_image_detection",
            device=device,
            # local_files_only=True
        )

        # 政治敏感标志检测（需自定义关键词）
        self.political_keywords = [
            'flag', 'symbol', 'propaganda',
            'leader', 'slogan', 'uniform'
        ]

        # # 暴恐内容检测模型
        # self.violence_detector = pipeline(
        #     "object-detection",
        #     model="felixwu/terror_detection",
        #     device=device
        # )

    def check_image(self, image_path):
        """执行完整检测流程"""
        try:
            img = Image.open(image_path).convert('RGB')
            results = {}

            # NSFW检测（色情/暴露内容）

            nsfw_result = self.nsfw_detector(img)
            # print(nsfw_result)
            results['nsfw'] = any(
                pred['score'] > 0.7 and pred['label'] in ['nsfw', 'porn']
                for pred in nsfw_result
            )

            # 暴恐内容检测
            # violence_objects = self.violence_detector(img)
            # results['violence'] = any(
            #     obj['score'] > 0.8
            #     for obj in violence_objects
            # )

            # 政治敏感检测（示例：简单OCR+关键词匹配）
            if self._check_political_content(img):
                results['political'] = True

            return results

        except Exception as e:
            return {'error': str(e)}

    def _check_political_content(self, image):
        """简易政治敏感检测（需配合实际OCR服务）"""
        # 实际使用时应接入OCR服务如Tesseract或百度API
        # 这里是模拟实现
        return False


# 使用示例
checker = ImageSafetyChecker()
# result = checker.check_image("setu.jpeg")
result = checker.check_image("liuying.png")
print(result)  # 输出示例: {'nsfw': False, 'violence': True, 'political': False}