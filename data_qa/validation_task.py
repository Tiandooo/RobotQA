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
                # 检查关节数据 (joint_1到joint_7)
                if col.startswith('joint_') and col[6:].isdigit():
                    expected_unit, factor = self.unit_standards['joint_[1-7]']
                    # 实际检查逻辑应更复杂，这里简化为值范围检查
                    if (df[col].abs() > 6.28).any():  # 2π≈6.28，超过可能不是弧度制
                        unit_issues.append(f"{col}可能有单位问题（应为{expected_unit}）")

            elif filename.startswith("joint_torque"):
                if col.startswith('joint_') and col[6:].isdigit():
                    expected_unit, factor = self.unit_standards['joint_torque']
                    if (df[col].abs() > 100).any():  # 超出100纽姆
                        unit_issues.append(f"{col}可能有单位问题（应为{expected_unit}，数值过大）")

            # 检查其他已知列
            elif col in self.unit_standards:
                expected_unit, factor = self.unit_standards[col]
                # 这里可以添加更具体的单位检查逻辑

        return unit_issues

    def check_naming_convention(self):
        """检查文件命名规范性"""
        naming_issues = []
        present_files = set(os.listdir(self.input_path))

        # 检查必须存在的文件
        for file_type, files in self.expected_files.items():
            for f in files:
                if f not in present_files:
                    naming_issues.append(f"缺少必要文件: {f}")
                else:
                    present_files.remove(f)

        # 检查多余文件
        # for extra_file in present_files:
        #     # 允许的额外文件类型（如README）
        #     if not (extra_file.endswith('.md') or extra_file.startswith('.')):
        #         naming_issues.append(f"存在未定义的额外文件: {extra_file}")

        return naming_issues

    def process_image_directory(self, dir_name):
        """处理单个图像目录的内容审核"""
        dir_path = os.path.join(self.input_path, dir_name)
        if not os.path.isdir(dir_path):
            return []

        unsafe_images = []
        for img_file in os.listdir(dir_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_path, img_file)
                result = self.check_content_safety(img_path)
                if any(result.get(k, False) for k in ['nsfw', 'political']):
                    unsafe_images.append({
                        'path': img_path,
                        'issues': result
                    })
        return unsafe_images

    def validate_csv_file(self, filename):
        """验证单个CSV文件"""
        issues = []
        try:
            df = pd.read_csv(os.path.join(self.input_path, filename))

            # 检查基本结构
            if 'timestamp' not in df.columns:
                issues.append("缺少timestamp列")
            if 'frame_idx' not in df.columns:
                issues.append("缺少frame_idx列")

            # 检查单位制
            unit_issues = self.validate_units(df, filename)
            issues.extend(unit_issues)

            # 检查时间戳连续性
            if len(df) > 1:
                time_diff = np.diff(df['timestamp'])
                if (time_diff <= 0).any():
                    issues.append("时间戳非单调递增")
                avg_freq = 1 / np.mean(time_diff)
                if not (4.9 <= avg_freq <= 5.1):
                    issues.append(f"采样频率异常: {avg_freq:.2f}Hz")

        except Exception as e:
            issues.append(f"文件解析失败: {str(e)}")

        return {filename: issues}

    def calculate_score(self, validation_results):
        """根据验证结果计算评分并确定等级（更严格的标准）"""
        # 初始化问题计数器
        critical_issues = 0
        major_issues = 0
        minor_issues = 0

        # 1. 命名规范问题（视为严重问题）
        naming_issues = len(validation_results['naming_convention'])
        critical_issues += naming_issues

        # 2. 内容安全问题（视为严重问题）
        content_issues = sum(len(v) for v in validation_results['content_safety'].values())
        critical_issues += content_issues

        # 3. 单位一致性问题（视情况分类）
        for filename, issues in validation_results['unit_consistency'].items():
            for issue in issues:
                if "数值过大" in issue:  # 扭矩单位错误是严重问题
                    critical_issues += 1
                else:  # 其他单位问题视为主要问题
                    major_issues += 1

        # 4. 结构问题（分类处理）
        for filename, issues in validation_results['structural_issues'].items():
            for issue in issues:
                if "缺少timestamp列" in issue or "缺少frame_idx列" in issue:
                    critical_issues += 1  # 缺少关键列是严重问题
                elif "时间戳非单调递增" in issue:
                    major_issues += 1  # 时间戳问题是主要问题
                elif "采样频率异常" in issue:
                    minor_issues += 1  # 频率偏差是次要问题
                else:
                    major_issues += 1  # 其他结构问题是主要问题

        # 计算扣分（更严格的扣分标准）
        score = 100
        score -= critical_issues * 15  # 每个严重问题扣15分
        score -= major_issues * 8  # 每个主要问题扣8分
        score -= minor_issues * 3  # 每个次要问题扣3分
        score = max(0, score)  # 确保不低于0分

        # 确定等级（更严格的标准）
        if score >= 95 and critical_issues == 0:
            level = "A+ (优秀)"
        elif score >= 85 and critical_issues <= 1:
            level = "A (良好)"
        elif score >= 75:
            level = "B (合格)"
        elif score >= 60:
            level = "C (基本合格)"
        else:
            level = "D (不合格)"

        return score, level

    def run_validation(self):
        """执行完整验证流程"""
        validation_results = {
            'naming_convention': [],
            'content_safety': {},
            'unit_consistency': {},
            'structural_issues': {}
        }

        # 阶段1: 文件命名规范检查
        validation_results['naming_convention'] = self.check_naming_convention()

        # 阶段2: CSV文件验证
        with ThreadPoolExecutor(max_workers=4) as executor:
            csv_results = list(executor.map(self.validate_csv_file, self.expected_files['csv']))

        for result in csv_results:
            for filename, issues in result.items():
                validation_results['structural_issues'][filename] = [
                    i for i in issues if not i.startswith('可能单位问题')
                ]
                validation_results['unit_consistency'][filename] = [
                    i for i in issues if i.startswith('可能单位问题')
                ]

        # 阶段3: 图像内容安全审核
        with ThreadPoolExecutor(max_workers=2) as executor:
            image_results = list(executor.map(self.process_image_directory, self.expected_files['image_dirs']))

        for dir_name, unsafe_images in zip(self.expected_files['image_dirs'], image_results):
            if unsafe_images:
                validation_results['content_safety'][dir_name] = unsafe_images

        # 计算评分和等级
        score, level = self.calculate_score(validation_results)

        # 生成总结报告
        summary = {
            'passed': all(not v for v in validation_results.values()),
            'score': score,
            'level': level,
            'details': validation_results
        }

        # 如果指定了输出路径，保存详细报告
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            report_path = os.path.join(self.output_path, 'validation_report.json')
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2)

        return {
            "status": "completed",
            "result": summary
        }


def data_validation_task(params, progress_callback, stop_event: Event):
    """对外暴露的验证任务接口"""
    validator = DataValidator(params, progress_callback, stop_event)
    return validator.run_validation()


params = {}
params["input_path"] = r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0"
print(data_validation_task(params, None, None))