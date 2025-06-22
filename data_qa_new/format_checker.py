import json
import os
import re
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd

from data_qa_new.qa_base import BaseChecker
from data_qa_new.image_safety import ImageSafetyChecker

class FormatChecker(BaseChecker):
    """
    规范性检查器，用于检测图像、视频和运动学数据的格式是否符合标准要求
    """
    def __init__(self, dataset_root: str):
        """
        初始化规范性检查器
        参数:
            dataset_root: 数据集根目录路径
        """
        super().__init__(dataset_root)
        self.results = {
            "image_checks": [],
            "video_checks": [],
            "kinematic_checks": [],
            "dynamic_checks": [],
            "naming_checks": [],
            "safety_checks": []  # 新增安全检查结果
        }
        # 预定义的文件命名规则
        self.naming_rules = {
            "image": {
                "pattern": r"^\d{6}\.jpg$",  # 例如: 000123.jpg
                "description": "图像文件名应为6位数字+.jpg后缀"
            },
            "depth": {
                "pattern": r"^\d{6}\.png$",  # 例如: 000123.png
                "description": "深度图文件名应为6位数字+.png后缀"
            },
            "video": {
                "pattern": r"^[a-zA-Z0-9_]+\.mp4$",  # 例如: camera1.mp4
                "description": "视频文件名应为字母数字下划线+.mp4后缀"
            },
            "kinematic": {
                "pattern": r"^state_[a-zA-Z0-9_]+\.csv$",  # 例如: state_joint.csv
                "description": "运动学数据文件名应为state_+字母数字下划线+.csv后缀"
            },
            "dynamic": {
                "pattern": r"^action_[a-zA-Z0-9_]+\.csv$",  # 例如: action_joint.csv
                "description": "动力学数据文件名应为action_+字母数字下划线+.csv后缀"
            }
        }
        
        # 存储文件路径的数据结构
        self.file_paths = {
            "images": {},  # 格式: {episode: {view: [file_paths]}}
            "depth": {},   # 格式: {episode: [file_paths]}
            "videos": {},  # 格式: {episode: [file_paths]}
            "kinematic": {},  # 格式: {episode: [file_paths]}
            "dynamic": {}   # 格式: {episode: [file_paths]}
        }
        
        # 初始化图像安全检查器
        self.safety_checker = ImageSafetyChecker()
        
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
                
                # 扫描视频数据
                videos_path = os.path.join(episode_path, "videos")
                if os.path.exists(videos_path):
                    mp4_files = [os.path.join(videos_path, f) for f in os.listdir(videos_path) if f.endswith('.mp4')]
                    self.file_paths["videos"][episode] = mp4_files
        
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
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['videos'])} 个视频目录")
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['kinematic'])} 个运动学数据目录")
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['dynamic'])} 个动力学数据目录")

    def check(self) -> Dict[str, Any]:
        """
        运行所有规范性检查
        返回:
            包含所有检查结果的字典
        """
        self.logger.info(f"开始检查数据集规范性: {self.dataset_root}")
        # 检查图像数据
        self._check_image_data()
        # 检查视频数据 (如果有)
        self._check_video_data()
        # 检查运动学数据
        self._check_kinematic_data()
        # 检查动力学数据
        self._check_dynamic_data()
        # 检查文件命名规则
        self._check_naming_rules()
        # 检查图像安全性
        self._check_image_safety()
        return self.results
        
    def _check_image_data(self) -> None:
        """
        检查图像数据格式
        """
        self.logger.info("开始检查图像数据格式...")
        
        # 检查RGB图像
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                for img_path in img_paths:
                    self._check_single_image(img_path, episode, view)
        
        # 检查深度图
        for episode, img_paths in self.file_paths["depth"].items():
            for img_path in img_paths:
                self._check_single_image(img_path, episode, "depth")
                
    def _check_single_image(self, img_path: str, episode: str, view: str) -> None:
        """
        检查单个图像文件格式
        参数:
            img_path: 图像文件路径
            episode: 所属episode
            view: 视角名称
        """
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                self.logger.error(f"无法读取图像文件: {img_path}")
                self.results["image_checks"].append({
                    "file": img_path,
                    "episode": episode,
                    "view": view,
                    "status": "error",
                    "message": "无法读取图像文件"
                })
                return
            # 检查数据类型
            dtype = img.dtype
            is_uint8 = dtype == np.uint8
            # 获取图像信息
            height, width, channels = img.shape
            # 记录结果
            self.results["image_checks"].append({
                "file": img_path,
                "episode": episode,
                "view": view,
                "status": "pass" if is_uint8 else "fail",
                "dtype": str(dtype),
                "expected_dtype": "uint8",
                "resolution": f"{width}x{height}",
                "channels": channels,
                "message": "像素数据类型符合要求" if is_uint8 else "像素数据类型不符合要求"
            })
            if not is_uint8:
                self.logger.warning(f"图像数据类型不符合要求: {img_path}, 类型: {dtype}, 应为: uint8")
        except Exception as e:
            self.logger.error(f"检查图像文件时出错: {img_path}, 错误: {str(e)}")
            self.results["image_checks"].append({
                "file": img_path,
                "episode": episode,
                "view": view,
                "status": "error",
                "message": f"检查时出错: {str(e)}"
            })
            
    def _check_video_data(self) -> None:
        """
        检查视频数据格式 (如果有)
        """
        self.logger.info("开始检查视频数据格式...")
        # 如果没有找到视频目录，记录信息并返回
        if not self.file_paths["videos"]:
            self.logger.info("未找到视频数据目录")
            return
        
        # 遍历所有视频文件
        for episode, video_paths in self.file_paths["videos"].items():
            # 随机抽样检查，最多检查5个文件
            sample_size = min(5, len(video_paths))
            if sample_size > 0:
                sample_files = np.random.choice(video_paths, sample_size, replace=False)
                
                for video_path in sample_files:
                    self._check_single_video(video_path)
                    
    def _check_single_video(self, video_path: str) -> None:
        """
        检查单个视频文件格式
        参数:
            video_path: 视频文件路径
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                self.results["video_checks"].append({
                    "file": video_path,
                    "status": "error",
                    "message": "无法打开视频文件"
                })
                return
            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 检查前5帧的数据类型
            frames_to_check = min(5, frame_count)
            frame_dtypes = []
            for i in range(frames_to_check):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_dtypes.append(str(frame.dtype))
            # 检查所有帧的数据类型是否一致且为uint8
            all_uint8 = all(dtype == "uint8" for dtype in frame_dtypes)
            all_consistent = len(set(frame_dtypes)) <= 1
            # 释放视频对象
            cap.release()
            # 记录结果
            self.results["video_checks"].append({
                "file": video_path,
                "status": "pass" if (all_uint8 and all_consistent) else "fail",
                "resolution": f"{width}x{height}",
                "fps": fps,
                "frame_count": frame_count,
                "frame_dtypes": frame_dtypes,
                "all_uint8": all_uint8,
                "all_consistent": all_consistent,
                "message": "视频帧数据类型符合要求" if (all_uint8 and all_consistent) else "视频帧数据类型不符合要求"
            })
            if not all_uint8:
                self.logger.warning(f"视频帧数据类型不符合要求: {video_path}, 应为: uint8")
            if not all_consistent:
                self.logger.warning(f"视频帧数据类型不一致: {video_path}")
        except Exception as e:
            self.logger.error(f"检查视频文件时出错: {video_path}, 错误: {str(e)}")
            self.results["video_checks"].append({
                "file": video_path,
                "status": "error",
                "message": f"检查时出错: {str(e)}"
            })
            
    def _check_kinematic_data(self) -> None:
        """
        检查运动学数据格式
        """
        self.logger.info("开始检查运动学数据格式...")
        
        # 如果没有找到运动学数据，记录信息并返回
        if not self.file_paths["kinematic"]:
            self.logger.warning("未找到运动学数据文件")
            return
        
        # 遍历所有运动学数据文件
        for episode, csv_paths in self.file_paths["kinematic"].items():
            for csv_path in csv_paths:
                self._check_single_csv(csv_path, episode, is_kinematic=True)
                
    def _check_dynamic_data(self) -> None:
        """
        检查动力学数据格式
        """
        self.logger.info("开始检查动力学数据格式...")
        
        # 如果没有找到动力学数据，记录信息并返回
        if not self.file_paths["dynamic"]:
            self.logger.warning("未找到动力学数据文件")
            return
        
        # 遍历所有动力学数据文件
        for episode, csv_paths in self.file_paths["dynamic"].items():
            for csv_path in csv_paths:
                self._check_single_csv(csv_path, episode, is_kinematic=False)
                
    def _check_single_csv(self, csv_path: str, episode: str, is_kinematic: bool) -> None:
        """
        检查单个CSV文件格式
        参数:
            csv_path: CSV文件路径
            episode: 所属episode
            is_kinematic: 是否为运动学数据
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            # 检查数据类型
            numeric_columns = df.select_dtypes(include=['number']).columns
            column_dtypes = {col: str(df[col].dtype) for col in numeric_columns}
            # 检查是否所有数值列都是float64
            all_float64 = all(dtype == 'float64' for col, dtype in column_dtypes.items() if col != 'frame_index')
            # 记录不符合要求的列
            non_float64_columns = [col for col, dtype in column_dtypes.items() if dtype != 'float64' and col != 'frame_index']
            # 记录结果
            result = {
                "file": csv_path,
                "episode": episode,
                "status": "pass" if all_float64 else "fail",
                "column_count": len(df.columns),
                "row_count": len(df),
                "all_float64": all_float64,
                "non_float64_columns": non_float64_columns,
                "message": "数据类型符合要求" if all_float64 else f"数据类型不符合要求: {len(non_float64_columns)}列不是float64"
            }
            if is_kinematic:
                self.results["kinematic_checks"].append(result)
            else:
                self.results["dynamic_checks"].append(result)
            if not all_float64:
                self.logger.warning(f"{'运动学' if is_kinematic else '动力学'}数据类型不符合要求: {csv_path}")
                self.logger.warning(f"不符合要求的列: {non_float64_columns}")
        except Exception as e:
            self.logger.error(f"检查CSV文件时出错: {csv_path}, 错误: {str(e)}")
            result = {
                "file": csv_path,
                "episode": episode,
                "status": "error",
                "message": f"检查时出错: {str(e)}"
            }
            if is_kinematic:
                self.results["kinematic_checks"].append(result)
            else:
                self.results["dynamic_checks"].append(result)

    def _check_naming_rules(self) -> None:
        """
        检查文件命名规则
        """
        self.logger.info("开始检查文件命名规则...")
        
        # 检查图像文件命名
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                for img_path in img_paths:
                    filename = os.path.basename(img_path)
                    pattern = self.naming_rules["image"]["pattern"]
                    match = re.match(pattern, filename)
                    if not match:
                        self.results["naming_checks"].append({
                            "file": img_path,
                            "type": "image",
                            "status": "fail",
                            "expected_pattern": pattern,
                            "message": f"图像文件名不符合规则: {filename}"
                        })
        
        # 检查深度图文件命名
        for episode, img_paths in self.file_paths["depth"].items():
            for img_path in img_paths:
                filename = os.path.basename(img_path)
                pattern = self.naming_rules["depth"]["pattern"]
                match = re.match(pattern, filename)
                if not match:
                    self.results["naming_checks"].append({
                        "file": img_path,
                        "type": "depth",
                        "status": "fail",
                        "expected_pattern": pattern,
                        "message": f"深度图文件名不符合规则: {filename}"
                    })

        # 检查视频文件命名
        for episode, video_paths in self.file_paths["videos"].items():
            for video_path in video_paths:
                filename = os.path.basename(video_path)
                pattern = self.naming_rules["video"]["pattern"]
                match = re.match(pattern, filename)
                if not match:
                    self.results["naming_checks"].append({
                        "file": video_path,
                        "type": "video",
                        "status": "fail",
                        "expected_pattern": pattern,
                        "message": f"视频文件名不符合规则: {filename}"
                    })
        
        # 检查运动学数据文件命名
        for episode, csv_paths in self.file_paths["kinematic"].items():
            for csv_path in csv_paths:
                filename = os.path.basename(csv_path)
                pattern = self.naming_rules["kinematic"]["pattern"]
                match = re.match(pattern, filename)
                if not match:
                    self.results["naming_checks"].append({
                        "file": csv_path,
                        "type": "kinematic",
                        "status": "fail",
                        "expected_pattern": pattern,
                        "message": f"运动学数据文件名不符合规则: {filename}"
                    })
        
        # 检查动力学数据文件命名
        for episode, csv_paths in self.file_paths["dynamic"].items():
            for csv_path in csv_paths:
                filename = os.path.basename(csv_path)
                pattern = self.naming_rules["dynamic"]["pattern"]
                match = re.match(pattern, filename)
                if not match:
                    self.results["naming_checks"].append({
                        "file": csv_path,
                        "type": "dynamic",
                        "status": "fail",
                        "expected_pattern": pattern,
                        "message": f"动力学数据文件名不符合规则: {filename}"
                    })
    
    def _check_image_safety(self) -> None:
        """
        检查图像安全性
        """
        self.logger.info("开始检查图像安全性...")
        
        # 随机抽样检查图像安全性，每个视角最多检查5张图片
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                sample_size = min(5, len(img_paths))
                if sample_size > 0:
                    sample_files = np.random.choice(img_paths, sample_size, replace=False)
                    
                    for img_path in sample_files:
                        img_path = str(img_path)
                        try:
                            # 使用图像安全检查器检查图像
                            safety_result = self.safety_checker.check_image(img_path)
                            
                            # 判断图像是否安全
                            is_safe = not (safety_result.get('nsfw', False) or 
                                          safety_result.get('political', False) or 
                                          safety_result.get('sensitive_words', False))
                            
                            # 记录结果
                            self.results["safety_checks"].append({
                                "file": img_path,
                                "episode": episode,
                                "view": view,
                                "status": "pass" if is_safe else "fail",
                                "nsfw": safety_result.get('nsfw', False),
                                "political": safety_result.get('political', False),
                                "sensitive_words": safety_result.get('sensitive_words', False),
                                "detected_words": safety_result.get('detected_words', []),
                                "message": "图像内容安全" if is_safe else "图像内容不安全"
                            })
                            
                            if not is_safe:
                                self.logger.warning(f"图像内容不安全: {img_path}")
                                if safety_result.get('nsfw', False):
                                    self.logger.warning(f"图像包含不适宜内容")
                                if safety_result.get('political', False):
                                    self.logger.warning(f"图像包含政治敏感内容")
                                if safety_result.get('sensitive_words', False):
                                    self.logger.warning(f"图像包含敏感词: {safety_result.get('detected_words', [])}")
                        except Exception as e:
                            self.logger.error(f"检查图像安全性时出错: {img_path}, 错误: {str(e)}")
                            self.results["safety_checks"].append({
                                "file": img_path,
                                "episode": episode,
                                "view": view,
                                "status": "error",
                                "message": f"检查时出错: {str(e)}"
                            })

    def generate_report(self, output_path: str = "format_check_report.json") -> None:
        """
        生成规范性检查报告
        参数:
            output_path: 输出报告的路径
        """
        # 计算统计信息
        stats = {
            "image": {
                "total": len(self.results["image_checks"]),
                "pass": sum(1 for check in self.results["image_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["image_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["image_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["image_checks"] if check["status"] == "pass") / len(self.results["image_checks"]) if self.results["image_checks"] else 0
            },
            "video": {
                "total": len(self.results["video_checks"]),
                "pass": sum(1 for check in self.results["video_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["video_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["video_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["video_checks"] if check["status"] == "pass") / len(self.results["video_checks"]) if self.results["video_checks"] else 0
            },
            "kinematic": {
                "total": len(self.results["kinematic_checks"]),
                "pass": sum(1 for check in self.results["kinematic_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["kinematic_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["kinematic_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["kinematic_checks"] if check["status"] == "pass") / len(self.results["kinematic_checks"]) if self.results["kinematic_checks"] else 0
            },
            "dynamic": {
                "total": len(self.results["dynamic_checks"]),
                "pass": sum(1 for check in self.results["dynamic_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["dynamic_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["dynamic_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["dynamic_checks"] if check["status"] == "pass") / len(self.results["dynamic_checks"]) if self.results["dynamic_checks"] else 0
            },
            "naming": {
                "total": len(self.results["naming_checks"]),
                "pass": sum(1 for check in self.results["naming_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["naming_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["naming_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["naming_checks"] if check["status"] == "pass") / len(self.results["naming_checks"]) if self.results["naming_checks"] else 0
            },
            "safety": {
                "total": len(self.results["safety_checks"]),
                "pass": sum(1 for check in self.results["safety_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["safety_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["safety_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["safety_checks"] if check["status"] == "pass") / len(self.results["safety_checks"]) if self.results["safety_checks"] else 0
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
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"规范性检查报告已保存至: {output_path}")
        # 打印摘要
        self.logger.info("规范性检查摘要:")
        self.logger.info(f"图像检查: 总计 {stats['image']['total']}, 通过 {stats['image']['pass']}, 失败 {stats['image']['fail']}, 错误 {stats['image']['error']}")
        self.logger.info(f"视频检查: 总计 {stats['video']['total']}, 通过 {stats['video']['pass']}, 失败 {stats['video']['fail']}, 错误 {stats['video']['error']}")
        self.logger.info(f"运动学数据检查: 总计 {stats['kinematic']['total']}, 通过 {stats['kinematic']['pass']}, 失败 {stats['kinematic']['fail']}, 错误 {stats['kinematic']['error']}")
        self.logger.info(f"动力学数据检查: 总计 {stats['dynamic']['total']}, 通过 {stats['dynamic']['pass']}, 失败 {stats['dynamic']['fail']}, 错误 {stats['dynamic']['error']}")
        self.logger.info(f"命名规则检查: 总计不符合规则 {stats['naming']['fail']}")
        self.logger.info(f"图像安全检查: 总计 {stats['safety']['total']}, 通过 {stats['safety']['pass']}, 失败 {stats['safety']['fail']}, 错误 {stats['safety']['error']}")

if __name__ == "__main__":
    dataset_root = "data/agibot_world/agibot_world_raw/356"
    output_dir = "reports"
    print(f"开始运行规范性检查...")
    format_checker = FormatChecker(dataset_root)
    format_results = format_checker.check()
    format_checker.generate_report(os.path.join(output_dir, "format_check_report.json"))
