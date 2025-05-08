import json
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from threading import Event
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
# import tensorflow as tf
from image_safety import ImageSafetyChecker


class DataValidator:
    def __init__(self, params, progress_callback, stop_event):
        self.input_path = params["input_path"]
        self.output_path = params.get("output_path", None)
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.content_check_model = self._load_content_check_model()
        self.checker = ImageSafetyChecker()
        self.current_date = datetime.now()

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

        # 数据标准配置
        self.data_standards = {
            'image': {
                'resolution': (640, 480),  # 标准分辨率
                'fps': 5  # 标准帧率
            },
            'cartesian_position': {
                'dimensions': ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
                'dtype': 'float64'
            },
            'joint_torque': {
                'dimensions': ['joint_1', 'joint_2', 'joint_3',
                               'joint_4', 'joint_5', 'joint_6'],
                'dtype': 'float64'
            }
        }

        # 单位制规范
        self.unit_standards = {
            'x': ('m', 1.0), 'y': ('m', 1.0), 'z': ('m', 1.0),
            'joint_[1-6]': ('rad', 1.0),
            'gripper_position': ('m', 1.0),
            'joint_torque': ('Nm', 1.0),
            'qx': ('unit', 1.0), 'qy': ('unit', 1.0),
            'qz': ('unit', 1.0), 'qw': ('unit', 1.0)
        }

        # 异常值阈值
        self.value_thresholds = {
            'joint_torque': 100,  # Nm
            'joint_position': 6.28,  # rad
            'cartesian_position': {
                'x': (-2, 2),  # m
                'y': (-2, 2),  # m
                'z': (0, 2)  # m
            }
        }

    def _load_content_check_model(self):
        """加载预训练的内容审核模型"""
        try:
            # model = tf.keras.models.load_model('path/to/model.h5')
            return None
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

    def process_image_directory(self, dir_name):
        """处理图像目录进行内容审核"""
        dir_path = os.path.join(self.input_path, dir_name)
        if not os.path.isdir(dir_path):
            return []

        unsafe_images = []
        for img_file in os.listdir(dir_path):
            if self.stop_event and self.stop_event.is_set():
                break

            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_path, img_file)
                result = self.check_content_safety(img_path)
                if any(result.get(k, False) for k in ['nsfw', 'political']):
                    unsafe_images.append({
                        'path': img_path,
                        'issues': result
                    })
        return unsafe_images

    def validate_units(self, df, filename):
        """检查数据单位制规范性"""
        unit_issues = []
        for col in df.columns:
            if col in ['frame_idx', 'timestamp']:
                continue

            # 检查关节扭矩数据
            if filename.startswith("joint_torque"):
                if col.startswith('joint_') and col[6:].isdigit():
                    expected_unit, factor = self.unit_standards['joint_torque']
                    if (df[col].abs() > self.value_thresholds['joint_torque']).any():
                        unit_issues.append(f"{col}单位可能错误（应为{expected_unit}）")

            # 检查笛卡尔位置数据
            elif filename.startswith("cartesian_position"):
                if col in self.unit_standards:
                    expected_unit, factor = self.unit_standards[col]
                    if col in ['x', 'y', 'z']:
                        min_val, max_val = self.value_thresholds['cartesian_position'][col]
                        if (df[col] < min_val).any() or (df[col] > max_val).any():
                            unit_issues.append(f"{col}值超出合理范围（{min_val}-{max_val}{expected_unit}）")

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
        for extra_file in present_files:
            if not (extra_file.endswith('.md') or extra_file.startswith('.')):
                naming_issues.append(f"存在未定义的额外文件: {extra_file}")

        return naming_issues

    def check_image_standards(self, dir_name):
        """检查图像分辨率、帧率是否符合标准"""
        standards_issues = []
        dir_path = os.path.join(self.input_path, dir_name)

        if not os.path.isdir(dir_path):
            return standards_issues

        # 检查图像分辨率
        for img_file in os.listdir(dir_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        if img.size != self.data_standards['image']['resolution']:
                            standards_issues.append(
                                f"{img_file}分辨率不符: {img.size} (应为{self.data_standards['image']['resolution']})"
                            )
                except Exception as e:
                    standards_issues.append(f"{img_file}无法打开: {str(e)}")

        # 检查视频帧率
        video_file = f"{dir_name}.mp4"
        video_path = os.path.join(self.input_path, video_file)
        if os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if abs(fps - self.data_standards['image']['fps']) > 0.5:
                    standards_issues.append(
                        f"{video_file}帧率不符: {fps:.1f}fps (应为{self.data_standards['image']['fps']}fps)"
                    )
                cap.release()
            except Exception as e:
                standards_issues.append(f"{video_file}无法读取: {str(e)}")

        return standards_issues

    def validate_data_structure(self, filename):
        """验证数据格式符合性"""
        structural_issues = []
        try:
            df = pd.read_csv(os.path.join(self.input_path, filename))

            # 检查基本结构
            if 'timestamp' not in df.columns:
                structural_issues.append("缺少timestamp列")
            if 'frame_idx' not in df.columns:
                structural_issues.append("缺少frame_idx列")

            # 检查特定文件类型的结构
            if filename.startswith("cartesian_position"):
                expected_dims = self.data_standards['cartesian_position']['dimensions']
                for dim in expected_dims:
                    if dim not in df.columns:
                        structural_issues.append(f"缺少必要维度: {dim}")

                # 检查数据类型
                if not all(df[expected_dims].dtypes == np.float64):
                    structural_issues.append("数据类型不符(应为float64)")

            elif filename.startswith("joint_torque"):
                expected_dims = self.data_standards['joint_torque']['dimensions']
                for dim in expected_dims:
                    if dim not in df.columns:
                        structural_issues.append(f"缺少必要维度: {dim}")

                if not all(df[expected_dims].dtypes == np.float64):
                    structural_issues.append("数据类型不符(应为float64)")

        except Exception as e:
            structural_issues.append(f"文件解析失败: {str(e)}")

        return structural_issues

    def check_data_completeness(self):
        """检查数据完整性"""
        completeness_issues = []

        # 检查CSV文件完整性
        for csv_file in self.expected_files['csv']:
            csv_path = os.path.join(self.input_path, csv_file)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # 检查时间步长是否连续
                    if len(df) > 1:
                        time_diff = np.diff(df['timestamp'])
                        if (time_diff <= 0).any():
                            completeness_issues.append(f"{csv_file}时间戳非单调递增")

                        # 检查丢帧情况
                        frame_diff = np.diff(df['frame_idx'])
                        if not all(frame_diff == 1):
                            completeness_issues.append(f"{csv_file}存在帧丢失")
                except Exception as e:
                    completeness_issues.append(f"{csv_file}解析失败: {str(e)}")

        # 检查图像-数据对齐
        if all(f in os.listdir(self.input_path) for f in ['cartesian_position.csv', 'exterior_image_1']):
            try:
                df = pd.read_csv(os.path.join(self.input_path, 'cartesian_position.csv'))
                image_dir = os.path.join(self.input_path, 'exterior_image_1')
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # 检查帧数是否匹配
                if len(df) != len(image_files):
                    completeness_issues.append(
                        f"图像与运动学数据帧数不匹配(数据:{len(df)}帧, 图像:{len(image_files)}帧)"
                    )
            except Exception as e:
                completeness_issues.append(f"图像-数据对齐检查失败: {str(e)}")

        return completeness_issues

    def check_data_accuracy(self):
        """检查数据准确性"""
        accuracy_issues = []

        for csv_file in self.expected_files['csv']:
            csv_path = os.path.join(self.input_path, csv_file)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)

                    # 检查缺失值
                    if df.isnull().any().any():
                        accuracy_issues.append(f"{csv_file}包含缺失值")

                    # 检查异常值
                    if csv_file.startswith("joint_torque"):
                        for col in df.columns:
                            if col.startswith('joint_') and col[6:].isdigit():
                                if (df[col].abs() > self.value_thresholds['joint_torque']).any():
                                    accuracy_issues.append(f"{csv_file}中{col}扭矩值异常")

                    elif csv_file.startswith("cartesian_position"):
                        for col in ['x', 'y', 'z']:
                            if col in df.columns:
                                min_val, max_val = self.value_thresholds['cartesian_position'][col]
                                if (df[col] < min_val).any() or (df[col] > max_val).any():
                                    accuracy_issues.append(f"{csv_file}中{col}位置值异常({min_val}-{max_val})")

                except Exception as e:
                    accuracy_issues.append(f"{csv_file}解析失败: {str(e)}")

        # 检查图像重复性
        for image_dir in self.expected_files['image_dirs']:
            dir_path = os.path.join(self.input_path, image_dir)
            if os.path.isdir(dir_path):
                image_hashes = {}
                for img_file in os.listdir(dir_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(dir_path, img_file)
                        try:
                            with Image.open(img_path) as img:
                                img_hash = hash(img.tobytes())
                                if img_hash in image_hashes:
                                    accuracy_issues.append(
                                        f"重复图像: {img_file} 和 {image_hashes[img_hash]}"
                                    )
                                else:
                                    image_hashes[img_hash] = img_file
                        except Exception as e:
                            accuracy_issues.append(f"{img_file}无法打开: {str(e)}")

        return accuracy_issues

    def check_data_freshness(self):
        """检查数据时效性"""
        freshness_issues = []

        # 获取数据创建时间
        creation_times = []
        for root, _, files in os.walk(self.input_path):
            for file in files:
                file_path = os.path.join(root, file)
                creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                creation_times.append(creation_time)

        if creation_times:
            oldest_date = min(creation_times)
            days_old = (self.current_date - oldest_date).days

            # 计算时间衰减指数 (TDI)
            tdi = max(0, 100 - days_old)  # 简单线性衰减模型

            if tdi < 60:
                freshness_issues.append(
                    f"数据时效性低(TDI={tdi:.1f}, 最旧数据: {oldest_date.strftime('%Y-%m-%d')})"
                )

            return freshness_issues, tdi

        return freshness_issues, 0

    def check_data_accessibility(self):
        """检查数据可访问性"""
        accessibility_issues = []

        # 检查所有文件可访问性
        for root, _, files in os.walk(self.input_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.lower().endswith('.csv'):
                        pd.read_csv(file_path, nrows=1)
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        Image.open(file_path).verify()
                    elif file.lower().endswith('.mp4'):
                        cap = cv2.VideoCapture(file_path)
                        if not cap.isOpened():
                            raise ValueError("无法打开视频文件")
                        cap.release()
                except Exception as e:
                    accessibility_issues.append(f"{file}无法访问: {str(e)}")

        return accessibility_issues

    def calculate_scores(self, validation_results):
        """计算各维度得分"""
        scores = {
            'data_norm': {
                'content_safety': {'score': 0, 'max': 20},
                'unit_consistency': {'score': 0, 'max': 20},
                'naming_convention': {'score': 0, 'max': 10}
            },
            'standard_compliance': {
                'format_compliance': {'score': 0, 'max': 20},
                'structure_compliance': {'score': 0, 'max': 10}
            },
            'data_completeness': {
                'trajectory_completeness': {'score': 0, 'max': 15},
                'visual_completeness': {'score': 0, 'max': 15}
            },
            'data_accuracy': {
                'value_accuracy': {'score': 0, 'max': 20},
                'image_accuracy': {'score': 0, 'max': 10}
            },
            'data_freshness': {
                'time_decay_index': {'score': 0, 'max': 10}
            },
            'data_accessibility': {
                'file_accessibility': {'score': 0, 'max': 10}
            }
        }

        # 1. 数据规范性评分
        # 内容安全性
        content_issues = sum(len(v) for v in validation_results['content_safety'].values())
        scores['data_norm']['content_safety']['score'] = max(0, 20 - content_issues * 2)

        # 单位一致性
        unit_issues = sum(len(v) for v in validation_results['unit_consistency'].values())
        scores['data_norm']['unit_consistency']['score'] = max(0, 20 - unit_issues * 2)

        # 命名规范性
        naming_issues = len(validation_results['naming_convention'])
        scores['data_norm']['naming_convention']['score'] = max(0, 10 - naming_issues * 2)

        # 2. 标准符合性评分
        # 格式符合性
        format_issues = sum(len(v) for v in validation_results['format_compliance'].values())
        scores['standard_compliance']['format_compliance']['score'] = max(0, 20 - format_issues * 2)

        # 结构符合性
        structure_issues = sum(len(v) for v in validation_results['structural_issues'].values())
        scores['standard_compliance']['structure_compliance']['score'] = max(0, 10 - structure_issues)

        # 3. 数据完整性评分
        # 轨迹完整性
        trajectory_issues = len([i for i in validation_results['completeness_issues']
                                 if "时间戳" in i or "帧丢失" in i])
        scores['data_completeness']['trajectory_completeness']['score'] = max(0, 15 - trajectory_issues * 3)

        # 视觉完整性
        visual_issues = len([i for i in validation_results['completeness_issues']
                             if "图像" in i or "帧数不匹配" in i])
        scores['data_completeness']['visual_completeness']['score'] = max(0, 15 - visual_issues * 3)

        # 4. 数据准确性评分
        # 数值准确性
        value_issues = len([i for i in validation_results['accuracy_issues']
                            if "缺失值" in i or "异常" in i])
        scores['data_accuracy']['value_accuracy']['score'] = max(0, 20 - value_issues * 2)

        # 图像准确性
        image_issues = len([i for i in validation_results['accuracy_issues']
                            if "重复图像" in i])
        scores['data_accuracy']['image_accuracy']['score'] = max(0, 10 - image_issues * 2)

        # 5. 数据时效性评分
        tdi = validation_results.get('time_decay_index', 0)
        scores['data_freshness']['time_decay_index']['score'] = min(tdi, 10)

        # 6. 数据可访问性评分
        accessibility_issues = len(validation_results['accessibility_issues'])
        scores['data_accessibility']['file_accessibility']['score'] = max(0, 10 - accessibility_issues * 2)

        # 计算总分和等级
        total_score = 0
        total_max = 0
        for category in scores.values():
            for metric in category.values():
                total_score += metric['score']
                total_max += metric['max']

        overall_score = round((total_score / total_max) * 100, 1) if total_max > 0 else 0

        # 确定等级
        if overall_score >= 90:
            level = "A+ (优秀)"
        elif overall_score >= 80:
            level = "A (良好)"
        elif overall_score >= 70:
            level = "B (合格)"
        elif overall_score >= 60:
            level = "C (基本合格)"
        else:
            level = "D (不合格)"

        return scores, overall_score, level

    def run_validation(self):
        """执行完整验证流程"""
        validation_results = {
            'naming_convention': [],
            'content_safety': {},
            'unit_consistency': {},
            'structural_issues': {},
            'format_compliance': {},
            'completeness_issues': [],
            'accuracy_issues': [],
            'accessibility_issues': [],
            'time_decay_index': 0
        }

        # 1. 数据规范性检查
        validation_results['naming_convention'] = self.check_naming_convention()

        # 内容安全审核
        with ThreadPoolExecutor(max_workers=2) as executor:
            image_results = list(executor.map(self.process_image_directory,
                                              self.expected_files['image_dirs']))

        for dir_name, unsafe_images in zip(self.expected_files['image_dirs'], image_results):
            if unsafe_images:
                validation_results['content_safety'][dir_name] = unsafe_images

        # 2. 标准符合性检查
        # CSV文件验证
        with ThreadPoolExecutor(max_workers=4) as executor:
            csv_results = list(executor.map(self.validate_data_structure, self.expected_files['csv']))

        for filename, issues in zip(self.expected_files['csv'], csv_results):
            validation_results['structural_issues'][filename] = issues
            df = pd.read_csv(os.path.join(self.input_path, filename))
            validation_results['unit_consistency'][filename] = self.validate_units(df, filename)

        # 图像标准检查
        with ThreadPoolExecutor(max_workers=2) as executor:
            format_results = list(executor.map(self.check_image_standards,
                                               self.expected_files['image_dirs']))

        for dir_name, issues in zip(self.expected_files['image_dirs'], format_results):
            validation_results['format_compliance'][dir_name] = issues

        # 3. 数据完整性检查
        validation_results['completeness_issues'] = self.check_data_completeness()

        # 4. 数据准确性检查
        validation_results['accuracy_issues'] = self.check_data_accuracy()

        # 5. 数据时效性检查
        freshness_issues, tdi = self.check_data_freshness()
        validation_results['completeness_issues'].extend(freshness_issues)
        validation_results['time_decay_index'] = tdi

        # 6. 数据可访问性检查
        validation_results['accessibility_issues'] = self.check_data_accessibility()

        # 计算评分
        scores, overall_score, level = self.calculate_scores(validation_results)

        # 生成总结报告
        summary = {
            'overall': {
                'score': overall_score,
                'level': level,
                'passed': all(not v for k, v in validation_results.items()
                              if k not in ['time_decay_index'])
            },
            'category_scores': scores,
            'details': validation_results
        }

        # 保存报告
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


# 使用示例
params = {
    "input_path": r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0",
    "output_path": r"D:\Astudy\RobotQA\reports"
}
print(data_validation_task(params, None, None))