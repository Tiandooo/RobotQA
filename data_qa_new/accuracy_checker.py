import hashlib
import os
from typing import Dict, Any, List, Set, Tuple

import numpy as np
import pandas as pd
import cv2

from data_qa_new.qa_base import BaseChecker

class AccuracyChecker(BaseChecker):
    """
    准确性检查器，用于检查数据集的准确性
    """
    def __init__(self, dataset_root: str):
        """
        初始化准确性检查器
        参数:
            dataset_root: 数据集根目录路径
        """
        super().__init__(dataset_root)
        self.results = {
            "image_hash_checks": [],
            "kinematic_value_checks": [],
            "dynamic_value_checks": []
        }
        
        # 存储文件路径的数据结构
        self.file_paths = {
            "images": {},  # 格式: {episode: {view: [file_paths]}}
            "depth": {},   # 格式: {episode: [file_paths]}
            "videos": {},  # 格式: {episode: [file_paths]}
            "kinematic": {},  # 格式: {episode: [file_paths]}
            "dynamic": {}   # 格式: {episode: [file_paths]}
        }
        
        # 存储图像哈希值的字典
        self.image_hashes = {}  # 格式: {hash_value: [file_paths]}
        
        # 运动学和动力学数据的合理阈值范围
        self.kinematic_thresholds = {
            # 关节角度阈值 (弧度)
            "joint_angle": (-3.14, 3.14),
            # 关节速度阈值 (弧度/秒)
            "joint_velocity": (-5.0, 5.0),
            # 关节加速度阈值 (弧度/秒^2)
            "joint_acceleration": (-10.0, 10.0),
            # 位置阈值 (米)
            "position": (-10.0, 10.0),
            # 速度阈值 (米/秒)
            "velocity": (-2.0, 2.0),
            # 加速度阈值 (米/秒^2)
            "acceleration": (-10.0, 10.0)
        }
        
        self.dynamic_thresholds = {
            # 力阈值 (牛顿)
            "force": (-100.0, 100.0),
            # 扭矩阈值 (牛米)
            "torque": (-50.0, 50.0),
            # 电流阈值 (安培)
            "current": (-20.0, 20.0),
            # 电压阈值 (伏特)
            "voltage": (-48.0, 48.0)
        }
        
        # 扫描目录结构
        self._scan_directory_structure()
        
    def _scan_directory_structure(self) -> None:
        """
        扫描目录结构，将文件路径保存到数据结构中
        """
        self.logger.info(f"扫描数据集目录结构: {self.dataset_root}")
        
        # 扫描图像和深度图数据
        obs_path = os.path.join(self.dataset_root, "observations")
        if os.path.exists(obs_path):
            for episode in os.listdir(obs_path):
                episode_path = os.path.join(obs_path, episode)
                if not os.path.isdir(episode_path):
                    continue
                
                # 扫描图像数据
                images_path = os.path.join(episode_path, "images")
                if os.path.exists(images_path):
                    self.file_paths["images"][episode] = {}
                    for view_dir in os.listdir(images_path):
                        view_path = os.path.join(images_path, view_dir)
                        if os.path.isdir(view_path):
                            jpg_files = [os.path.join(view_path, f) for f in os.listdir(view_path) if f.endswith('.jpg')]
                            self.file_paths["images"][episode][view_dir] = jpg_files
                
                # 扫描深度图数据
                depth_path = os.path.join(episode_path, "depth")
                if os.path.exists(depth_path):
                    png_files = [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith('.png')]
                    self.file_paths["depth"][episode] = png_files
        
        # 扫描运动学和动力学数据
        prop_path = os.path.join(self.dataset_root, "proprio_stats")
        if os.path.exists(prop_path):
            for episode in os.listdir(prop_path):
                episode_path = os.path.join(prop_path, episode)
                if not os.path.isdir(episode_path):
                    continue
                
                # 获取所有CSV文件
                csv_files = [os.path.join(episode_path, f) for f in os.listdir(episode_path) if f.endswith('.csv')]
                
                # 分类CSV文件
                kinematic_files = [f for f in csv_files if os.path.basename(f).startswith('state_')]
                dynamic_files = [f for f in csv_files if os.path.basename(f).startswith('action_')]
                
                if kinematic_files:
                    self.file_paths["kinematic"][episode] = kinematic_files
                if dynamic_files:
                    self.file_paths["dynamic"][episode] = dynamic_files
        
        # 记录扫描结果
        self.logger.info(f"扫描完成，找到 {sum(len(views) for views in self.file_paths['images'].values())} 个图像视角目录")
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['depth'])} 个深度图目录")
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['kinematic'])} 个运动学数据目录")
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['dynamic'])} 个动力学数据目录")
    
    def check(self) -> Dict[str, Any]:
        """
        运行所有准确性检查
        返回:
            包含所有检查结果的字典
        """
        self.logger.info(f"开始检查数据集准确性: {self.dataset_root}")
        
        # 检查图像哈希值
        self._check_image_hashes()
        
        # 检查运动学数据的有效性
        self._check_kinematic_values()
        
        # 检查动力学数据的有效性
        self._check_dynamic_values()
        
        return self.results
    
    def _calculate_image_hash(self, img_path: str) -> str:
        """
        计算图像的MD5哈希值
        参数:
            img_path: 图像文件路径
        返回:
            图像的MD5哈希值
        """
        try:
            # 读取图像文件
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            # 计算MD5哈希值
            md5_hash = hashlib.md5(img_data).hexdigest()
            return md5_hash
        except Exception as e:
            self.logger.error(f"计算图像哈希值时出错: {img_path}, 错误: {str(e)}")
            return None
    
    def _check_image_hashes(self) -> None:
        """
        检查图像哈希值，检测重复图像
        """
        self.logger.info("开始检查图像哈希值...")
        
        # 计算所有图像的哈希值
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                for img_path in img_paths:
                    hash_value = self._calculate_image_hash(img_path)
                    if hash_value:
                        if hash_value not in self.image_hashes:
                            self.image_hashes[hash_value] = []
                        self.image_hashes[hash_value].append(img_path)
        
        # 检查是否存在重复图像
        duplicate_count = 0
        for hash_value, file_paths in self.image_hashes.items():
            if len(file_paths) > 1:
                duplicate_count += 1
                self.results["image_hash_checks"].append({
                    "hash": hash_value,
                    "files": file_paths,
                    "count": len(file_paths),
                    "status": "fail",
                    "message": f"发现{len(file_paths)}个重复图像"
                })
                self.logger.warning(f"发现重复图像: {file_paths}")
        
        self.logger.info(f"图像哈希值检查完成，共发现{duplicate_count}组重复图像")
    
    def _check_kinematic_values(self) -> None:
        """
        检查运动学数据的有效性
        """
        self.logger.info("开始检查运动学数据的有效性...")
        
        # 遍历所有运动学数据文件
        for episode, csv_paths in self.file_paths["kinematic"].items():
            for csv_path in csv_paths:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否存在空值
                    null_count = df.isnull().sum().sum()
                    has_null = null_count > 0
                    
                    # 检查是否存在无效值 (NaN, Inf)
                    invalid_count = np.isnan(df.values).sum() + np.isinf(df.values).sum()
                    has_invalid = invalid_count > 0
                    
                    # 检查数值是否在合理范围内
                    out_of_range_columns = {}
                    for col in df.columns:
                        if col == 'frame_index':
                            continue
                        
                        # 根据列名判断数据类型
                        threshold_key = None
                        for key in self.kinematic_thresholds.keys():
                            if key in col.lower():
                                threshold_key = key
                                break
                        
                        if threshold_key:
                            min_val, max_val = self.kinematic_thresholds[threshold_key]
                            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                            if out_of_range > 0:
                                out_of_range_columns[col] = {
                                    "threshold": (min_val, max_val),
                                    "min": df[col].min(),
                                    "max": df[col].max(),
                                    "out_of_range_count": out_of_range
                                }
                    
                    # 记录结果
                    self.results["kinematic_value_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "null_count": null_count,
                        "invalid_count": invalid_count,
                        "out_of_range_columns": out_of_range_columns,
                        "status": "pass" if (not has_null and not has_invalid and not out_of_range_columns) else "fail",
                        "message": "运动学数据有效" if (not has_null and not has_invalid and not out_of_range_columns) else "运动学数据存在问题"
                    })
                    
                    if has_null or has_invalid or out_of_range_columns:
                        self.logger.warning(f"运动学数据存在问题: {csv_path}")
                        if has_null:
                            self.logger.warning(f"  - 存在{null_count}个空值")
                        if has_invalid:
                            self.logger.warning(f"  - 存在{invalid_count}个无效值")
                        if out_of_range_columns:
                            self.logger.warning(f"  - {len(out_of_range_columns)}列数据超出合理范围")
                except Exception as e:
                    self.logger.error(f"检查运动学数据时出错: {csv_path}, 错误: {str(e)}")
                    self.results["kinematic_value_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def _check_dynamic_values(self) -> None:
        """
        检查动力学数据的有效性
        """
        self.logger.info("开始检查动力学数据的有效性...")
        
        # 遍历所有动力学数据文件
        for episode, csv_paths in self.file_paths["dynamic"].items():
            for csv_path in csv_paths:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否存在空值
                    null_count = df.isnull().sum().sum()
                    has_null = null_count > 0
                    
                    # 检查是否存在无效值 (NaN, Inf)
                    invalid_count = np.isnan(df.values).sum() + np.isinf(df.values).sum()
                    has_invalid = invalid_count > 0
                    
                    # 检查数值是否在合理范围内
                    out_of_range_columns = {}
                    for col in df.columns:
                        if col == 'frame_index':
                            continue
                        
                        # 根据列名判断数据类型
                        threshold_key = None
                        for key in self.dynamic_thresholds.keys():
                            if key in col.lower():
                                threshold_key = key
                                break
                        
                        if threshold_key:
                            min_val, max_val = self.dynamic_thresholds[threshold_key]
                            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                            if out_of_range > 0:
                                out_of_range_columns[col] = {
                                    "threshold": (min_val, max_val),
                                    "min": df[col].min(),
                                    "max": df[col].max(),
                                    "out_of_range_count": out_of_range
                                }
                    
                    # 记录结果
                    self.results["dynamic_value_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "null_count": null_count,
                        "invalid_count": invalid_count,
                        "out_of_range_columns": out_of_range_columns,
                        "status": "pass" if (not has_null and not has_invalid and not out_of_range_columns) else "fail",
                        "message": "动力学数据有效" if (not has_null and not has_invalid and not out_of_range_columns) else "动力学数据存在问题"
                    })
                    
                    if has_null or has_invalid or out_of_range_columns:
                        self.logger.warning(f"动力学数据存在问题: {csv_path}")
                        if has_null:
                            self.logger.warning(f"  - 存在{null_count}个空值")
                        if has_invalid:
                            self.logger.warning(f"  - 存在{invalid_count}个无效值")
                        if out_of_range_columns:
                            self.logger.warning(f"  - {len(out_of_range_columns)}列数据超出合理范围")
                except Exception as e:
                    self.logger.error(f"检查动力学数据时出错: {csv_path}, 错误: {str(e)}")
                    self.results["dynamic_value_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def generate_report(self, output_path: str = "accuracy_check_report.json") -> None:
        """
        生成准确性检查报告
        参数:
            output_path: 输出报告的路径
        """
        # 计算统计信息
        stats = {
            "image_hash": {
                "total_images": sum(len(file_paths) for file_paths in self.image_hashes.values()),
                "unique_images": len(self.image_hashes),
                "duplicate_groups": sum(1 for file_paths in self.image_hashes.values() if len(file_paths) > 1),
                "duplicate_images": sum(len(file_paths) - 1 for file_paths in self.image_hashes.values() if len(file_paths) > 1)
            },
            "kinematic_value": {
                "total": len(self.results["kinematic_value_checks"]),
                "pass": sum(1 for check in self.results["kinematic_value_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["kinematic_value_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["kinematic_value_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["kinematic_value_checks"] if check["status"] == "pass") / len(self.results["kinematic_value_checks"]) if self.results["kinematic_value_checks"] else 0
            },
            "dynamic_value": {
                "total": len(self.results["dynamic_value_checks"]),
                "pass": sum(1 for check in self.results["dynamic_value_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["dynamic_value_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["dynamic_value_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["dynamic_value_checks"] if check["status"] == "pass") / len(self.results["dynamic_value_checks"]) if self.results["dynamic_value_checks"] else 0
            }
        }
        
        # 生成报告
        report = {
            "dataset_root": self.dataset_root,
            "timestamp": pd.Timestamp.now().isoformat(),
            "statistics": stats,
            "results": {key: value for key, value in self.results.items() if any(check["status"] != "pass" for check in value)}
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"准确性检查报告已保存至: {output_path}")
        
        # 打印摘要
        self.logger.info("准确性检查摘要:")
        self.logger.info(f"图像哈希值检查: 总计 {stats['image_hash']['total_images']} 张图像, 唯一 {stats['image_hash']['unique_images']} 张, 重复 {stats['image_hash']['duplicate_images']} 张 (分布在 {stats['image_hash']['duplicate_groups']} 组)")
        self.logger.info(f"运动学数据检查: 总计 {stats['kinematic_value']['total']}, 通过 {stats['kinematic_value']['pass']}, 失败 {stats['kinematic_value']['fail']}, 错误 {stats['kinematic_value']['error']}")
        self.logger.info(f"动力学数据检查: 总计 {stats['dynamic_value']['total']}, 通过 {stats['dynamic_value']['pass']}, 失败 {stats['dynamic_value']['fail']}, 错误 {stats['dynamic_value']['error']}")


if __name__ == "__main__":
    dataset_root = "data/agibot_world/agibot_world_raw/356"
    output_dir = "reports"
    print(f"开始运行准确性检查...")
    accuracy_checker = AccuracyChecker(dataset_root)
    accuracy_results = accuracy_checker.check()
    accuracy_checker.generate_report(os.path.join(output_dir, "accuracy_check_report.json"))

