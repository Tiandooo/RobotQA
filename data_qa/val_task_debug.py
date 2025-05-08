import json
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from threading import Event
import shutil
from concurrent.futures import ThreadPoolExecutor
# import tensorflow as tf  # 用于内容审核模型加载
from image_safety import ImageSafetyChecker


class DataValidator:
    def __init__(self, params, progress_callback, stop_event):
        self.input_path = params["input_path"]
        self.output_path = params.get("output_path", None)
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.content_check_model = self._load_content_check_model()
        self.checker = ImageSafetyChecker()

        # 预定义规范配置
        self.expected_files = {
            'csv': [
                'cartesian_position.csv',
                'cartesian_joint.csv',
                'joint_torque.csv',
                'gripper_position.csv'
            ],
            'image_dirs': [
                'exterior_image_1',
                'wrist_image_1'
            ],
            'video_files': [
                'exterior_image_1.mp4',
                'wrist_image_1.mp4'
            ]
        }

        # 单位制规范 (列名: (允许单位, 转换系数))
        self.unit_standards = {
            'x': ('m', 1.0), 'y': ('m', 1.0), 'z': ('m', 1.0),
            'joint_[1-7]': ('rad', 1.0),
            'gripper_position': ('m', 1.0),
            'joint_torque': ('Nm', 1.0)
        }

    def _load_content_check_model(self):
        """加载预训练的内容审核模型（示例结构）"""
        try:
            # 实际应用中替换为真实的模型加载代码
            model = tf.keras.models.load_model('path/to/content_check_model.h5')
            return model
        except Exception as e:
            print(f"Could not load content check model: {str(e)}")
            return None

    def check_content_safety(self, image_path):
        """图像内容安全检测"""
        try:
            result = self.checker.check_image(image_path)
            return result
        except Exception as e:
            print(f"Content check failed for {image_path}: {str(e)}")
            return {'error': str(e)}

    def validate_units(self, df, filename):
        """检查数据单位制规范性"""
        unit_issues = []
        for col in df.columns:
            if col in ['frame_idx', 'timestamp']:
                continue

            if filename.startswith("cartesian_joint"):
                if col.startswith('joint_') and col[6:].isdigit():
                    expected_unit, factor = self.unit_standards['joint_[1-7]']
                    if (df[col].abs() > 6.28).any():
                        unit_issues.append(f"{col}可能有单位问题（应为{expected_unit}）")

            elif filename.startswith("joint_torque"):
                if col.startswith('joint_') and col[6:].isdigit():
                    expected_unit, factor = self.unit_standards['joint_torque']
                    if (df[col].abs() > 100).any():
                        unit_issues.append(f"{col}可能有单位问题（应为{expected_unit}，数值过大）")

            elif col in self.unit_standards:
                expected_unit, factor = self.unit_standards[col]
                # 其他列单位检查逻辑可扩展

        return unit_issues

    def check_naming_convention(self):
        """检查文件命名规范性"""
        naming_issues = []
        present_files = set(os.listdir(self.input_path))

        for file_type, files in self.expected_files.items():
            for f in files:
                if f not in present_files:
                    naming_issues.append(f"缺少必要文件: {f}")
                else:
                    present_files.remove(f)
        return naming_issues

    def process_image_directory(self, dir_name):
        """处理单个图像目录的内容审核"""
        dir_path = os.path.join(self.input_path, dir_name)
        unsafe_images = []
        if os.path.isdir(dir_path):
            for img_file in os.listdir(dir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.input_path, dir_name, img_file)
                    result = self.check_content_safety(img_path)
                    if any(result.get(k, False) for k in ['nsfw', 'political']):
                        unsafe_images.append({'path': img_path, 'issues': result})
        return unsafe_images

    def validate_csv_file(self, filename):
        """验证单个CSV文件"""
        issues = []
        try:
            df = pd.read_csv(os.path.join(self.input_path, filename))
            if 'timestamp' not in df.columns:
                issues.append("缺少timestamp列")
            if 'frame_idx' not in df.columns:
                issues.append("缺少frame_idx列")

            unit_issues = self.validate_units(df, filename)
            issues.extend(unit_issues)

            if 'timestamp' in df.columns and len(df) > 1:
                time_diff = np.diff(df['timestamp'])
                if (time_diff <= 0).any():
                    issues.append("时间戳非单调递增")
                avg_freq = 1 / np.mean(time_diff)
                if not (4.9 <= avg_freq <= 5.1):
                    issues.append(f"采样频率异常: {avg_freq:.2f}Hz")

        except Exception as e:
            issues.append(f"文件解析失败: {str(e)}")

        return {filename: issues}

    def compute_section_level(self, issues_count, thresholds):
        """根据问题数量和阈值返回等级"""
        if issues_count <= thresholds['A']:
            return 'A (良好)'
        elif issues_count <= thresholds['B']:
            return 'B (合格)'
        elif issues_count <= thresholds['C']:
            return 'C (基本合格)'
        else:
            return 'D (不合格)'

    def run_validation(self):
        validation_results = {
            'section_1': [],
            'section_2': [],
            'section_3': [],
            'section_4': [],
            'section_6': [],
            'section_7': []
        }

        # Section 1: 数据规范性
        sc1_naming = self.check_naming_convention()
        validation_results['section_1'].extend(sc1_naming)
        for dir_name in self.expected_files['image_dirs']:
            unsafe = self.process_image_directory(dir_name)
            if unsafe:
                validation_results['section_1'].append(f"{dir_name} 内容安全问题 x {len(unsafe)}")

        # Section 2: 数据标准符合性
        for result in ThreadPoolExecutor(max_workers=4).map(self.validate_csv_file, self.expected_files['csv']):
            for fn, issues in result.items():
                if issues:
                    validation_results['section_2'].append((fn, issues))

        # Section 3 & 4: 数据完整性与准确性
        # 简化示例：把所有 CSV issues 都视为完整或准确性问题
        for fn, issues in validation_results['section_2']:
            validation_results['section_3'].append(issues)
            validation_results['section_4'].append(issues)

        # Section 6: 数据时效性 (占位)
        validation_results['section_6'].append('TDI 计算待实现')

        # Section 7: 数据可访问性
        # Section 7: 数据可访问性


        access_issues = []

        for root, dirs, files in os.walk(self.input_path):
            for f in files:
                f_path = os.path.join(root, f)
                try:
                    with open(f_path, 'rb'):
                        pass
                except Exception:
                    rel_path = os.path.relpath(f_path, self.input_path)  # 可选: 用相对路径，报告更清晰
                    access_issues.append(f"无法访问: {rel_path}")

        validation_results['section_7'].extend(access_issues)

        # 生成每个分项等级和说明
        section_thresholds = {
            'section_1': {'A': 0, 'B': 2, 'C': 5},
            'section_2': {'A': 0, 'B': 1, 'C': 5},
            'section_3': {'A': 0, 'B': 1, 'C': 5},
            'section_4': {'A': 0, 'B': 1, 'C': 5},
            'section_6': {'A': 0, 'B': 1, 'C': 2},
            'section_7': {'A': 0, 'B': 1, 'C': 2}
        }
        levels = {}
        explanations = {}
        for sec, items in validation_results.items():
            count = len(items)
            level = self.compute_section_level(count, section_thresholds[sec])
            levels[sec] = level
            explanations[sec] = f"共发现问题 {count} 项，依据阈值设定评为 {level}"

        return {
            'levels': levels,
            'explanations': explanations,
            'details': validation_results
        }


def data_validation_task(params, progress_callback, stop_event: Event):
    validator = DataValidator(params, progress_callback, stop_event)
    return validator.run_validation()

if __name__ == '__main__':
    params = {"input_path": r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0"}
    print(data_validation_task(params, None, None))
