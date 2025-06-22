import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline
import os
import easyocr
import re
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

        # 初始化OCR读取器
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())

        # 从文件读取敏感词列表
        self.sensitive_words = self._load_sensitive_words("敏感词汇.txt")

        # # 暴恐内容检测模型
        # self.violence_detector = pipeline(
        #     "object-detection",
        #     model="felixwu/terror_detection",
        #     device=device
        # )
    def _load_sensitive_words(self, file_path):
        """从文件中加载敏感词列表，要求文件第一行为 #KWLIST 标识"""
        sensitive_words = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取第一行并验证
                first_line = f.readline().strip()
                if first_line != "#KWLIST":
                    print(f"敏感词文件格式错误: 第一行应为 #KWLIST，实际为 {first_line}")
                    raise ValueError("文件格式错误")
                
                # 读取剩余行
                for line in f:
                    word = line.strip()
                    if word:  # 忽略空行
                        sensitive_words.append(word)
            print(f"成功加载 {len(sensitive_words)} 个敏感词")
        except Exception as e:
            print(f"加载敏感词文件失败: {str(e)}")
            # 提供一些默认敏感词作为备选
            sensitive_words = [
                '反动', '颠覆', '抗议', '示威', '游行', '革命',
                '杀人', '爆炸', '袭击', '恐怖', '暴力',
                '色情', '裸露', '性交', '卖淫',
            ]
            print(f"使用默认敏感词列表: {len(sensitive_words)} 个词")
        return sensitive_words

    def check_image(self, image_path):
        """执行完整检测流程"""
        img = Image.open(image_path).convert('RGB')
        results = {}

        # # NSFW检测（色情/暴露内容）
        nsfw_result = self.nsfw_detector(img)
        results['nsfw'] = any(
            pred['score'] > 0.7 and pred['label'] in ['nsfw', 'porn']
            for pred in nsfw_result
        )
        print(results['nsfw'])

        # 政治敏感检测（OCR+关键词匹配）
        political_result = self._check_political_content(img)
        results['political'] = political_result['detected']

        # 敏感词检测
        sensitive_result = self._check_sensitive_words(image_path)
        results['sensitive_words'] = sensitive_result['detected']
        if sensitive_result['detected']:
            results['detected_words'] = sensitive_result['words']
            results['text_content'] = sensitive_result['text']

        return results

    def _check_political_content(self, image):
        """简易政治敏感检测（需配合实际OCR服务）"""
        # 实际使用时应接入OCR服务如Tesseract或百度API
        # 这里是模拟实现
        return {'detected': False}

    def _check_sensitive_words(self, image_path):
        """使用OCR检测图片中的文字并匹配敏感词"""

        # 使用EasyOCR识别图片中的文字
        results = self.reader.readtext(image_path)

        # 提取所有识别出的文本
        all_text = ' '.join([text[1] for text in results])
        print(all_text)

        # 检查是否包含敏感词
        detected_words = []
        for word in self.sensitive_words:
            if re.search(word, all_text, re.IGNORECASE):
                detected_words.append(word)

        return {
            'detected': len(detected_words) > 0,
            'words': detected_words,
            'text': all_text
        }


if __name__ == "__main__":
    # 使用示例
    checker = ImageSafetyChecker()
    # result = checker.check_image("setu.jpeg")
    result = checker.check_image("img_3.png")
    print(result)  # 输出示例: {'nsfw': False, 'political': False, 'sensitive_words': True, 'detected_words': ['敏感词1']}
