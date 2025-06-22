import json
import os
from typing import Dict, Any, List, Set

import pandas as pd

from data_qa_new.qa_base import BaseChecker

class CompletenessChecker(BaseChecker):
    """
    完整性检查器，用于检查数据集的完整性
    """
    def __init__(self, dataset_root: str, metadata_file: str = None):
        """
        初始化完整性检查器
        参数:
            dataset_root: 数据集根目录路径
            metadata_file: 元数据描述文件路径，如果为None则自动查找
        """
        super().__init__(dataset_root)
        self.metadata_file = metadata_file
        self.metadata = None
        self.results = {
            "metadata_checks": [],
            "file_existence_checks": [],
            "frame_index_checks": []
        }
        
        # 存储文件路径的数据结构
        self.file_paths = {
            "images": {},  # 格式: {episode: {view: [file_paths]}}
            "depth": {},   # 格式: {episode: [file_paths]}
            "videos": {},  # 格式: {episode: [file_paths]}
            "kinematic": {},  # 格式: {episode: [file_paths]}
            "dynamic": {}   # 格式: {episode: [file_paths]}
        }
        
        # 扫描目录结构
        self._scan_directory_structure()
        
        # 加载元数据
        self._load_metadata()
        
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
        
    def _load_metadata(self) -> None:
        """
        加载元数据描述文件
        """
        # 如果未指定元数据文件，则尝试在数据集根目录下查找
        if self.metadata_file is None:
            possible_metadata_files = [
                os.path.join(self.dataset_root, "metadata.json"),
                os.path.join(self.dataset_root, "dataset_metadata.json"),
                os.path.join(self.dataset_root, "dataset_info.json")
            ]
            
            for file_path in possible_metadata_files:
                if os.path.exists(file_path):
                    self.metadata_file = file_path
                    break
        
        # 如果找到元数据文件，则加载
        if self.metadata_file and os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"成功加载元数据文件: {self.metadata_file}")
                
                # 记录元数据检查结果
                self.results["metadata_checks"].append({
                    "file": self.metadata_file,
                    "status": "pass",
                    "message": "成功加载元数据文件"
                })
            except Exception as e:
                self.logger.error(f"加载元数据文件时出错: {self.metadata_file}, 错误: {str(e)}")
                self.results["metadata_checks"].append({
                    "file": self.metadata_file,
                    "status": "error",
                    "message": f"加载元数据文件时出错: {str(e)}"
                })
        else:
            self.logger.warning("未找到元数据文件")
            self.results["metadata_checks"].append({
                "status": "warning",
                "message": "未找到元数据文件"
            })
    
    def check(self) -> Dict[str, Any]:
        """
        运行所有完整性检查
        返回:
            包含所有检查结果的字典
        """
        self.logger.info(f"开始检查数据集完整性: {self.dataset_root}")
        
        # 检查元数据中描述的文件是否存在
        if self.metadata:
            self._check_files_existence()
        
        # 检查图像文件名和CSV文件的frame_index时间戳完整性
        self._check_frame_index_completeness()
        
        return self.results
    
    def _check_files_existence(self) -> None:
        """
        检查元数据中描述的文件是否存在
        """
        self.logger.info("开始检查元数据中描述的文件是否存在...")
        
        # 检查元数据中的文件列表
        if 'files' in self.metadata:
            for file_info in self.metadata['files']:
                file_path = os.path.join(self.dataset_root, file_info.get('path', ''))
                file_exists = os.path.exists(file_path)
                
                self.results["file_existence_checks"].append({
                    "file": file_path,
                    "status": "pass" if file_exists else "fail",
                    "message": "文件存在" if file_exists else "文件不存在"
                })
                
                if not file_exists:
                    self.logger.warning(f"元数据中描述的文件不存在: {file_path}")
        
        # 检查元数据中的episodes列表
        if 'episodes' in self.metadata:
            for episode_info in self.metadata['episodes']:
                episode_id = episode_info.get('id', '')
                
                # 检查图像文件
                if 'images' in episode_info:
                    for view_info in episode_info['images']:
                        view_id = view_info.get('view', '')
                        image_count = view_info.get('count', 0)
                        
                        # 检查实际文件数量是否与元数据描述一致
                        actual_count = len(self.file_paths["images"].get(episode_id, {}).get(view_id, []))
                        is_complete = actual_count >= image_count
                        
                        self.results["file_existence_checks"].append({
                            "episode": episode_id,
                            "view": view_id,
                            "type": "image",
                            "expected_count": image_count,
                            "actual_count": actual_count,
                            "status": "pass" if is_complete else "fail",
                            "message": "图像文件数量符合要求" if is_complete else f"图像文件数量不足: 期望{image_count}，实际{actual_count}"
                        })
                        
                        if not is_complete:
                            self.logger.warning(f"图像文件数量不足: 期望{image_count}，实际{actual_count}，episode={episode_id}, view={view_id}")
                
                # 检查深度图文件
                if 'depth' in episode_info:
                    depth_count = episode_info['depth'].get('count', 0)
                    
                    # 检查实际文件数量是否与元数据描述一致
                    actual_count = len(self.file_paths["depth"].get(episode_id, []))
                    is_complete = actual_count >= depth_count
                    
                    self.results["file_existence_checks"].append({
                        "episode": episode_id,
                        "type": "depth",
                        "expected_count": depth_count,
                        "actual_count": actual_count,
                        "status": "pass" if is_complete else "fail",
                        "message": "深度图文件数量符合要求" if is_complete else f"深度图文件数量不足: 期望{depth_count}，实际{actual_count}"
                    })
                    
                    if not is_complete:
                        self.logger.warning(f"深度图文件数量不足: 期望{depth_count}，实际{actual_count}，episode={episode_id}")
                
                # 检查视频文件
                if 'videos' in episode_info:
                    video_count = episode_info['videos'].get('count', 0)
                    
                    # 检查实际文件数量是否与元数据描述一致
                    actual_count = len(self.file_paths["videos"].get(episode_id, []))
                    is_complete = actual_count >= video_count
                    
                    self.results["file_existence_checks"].append({
                        "episode": episode_id,
                        "type": "video",
                        "expected_count": video_count,
                        "actual_count": actual_count,
                        "status": "pass" if is_complete else "fail",
                        "message": "视频文件数量符合要求" if is_complete else f"视频文件数量不足: 期望{video_count}，实际{actual_count}"
                    })
                    
                    if not is_complete:
                        self.logger.warning(f"视频文件数量不足: 期望{video_count}，实际{actual_count}，episode={episode_id}")
                
                # 检查运动学数据文件
                if 'kinematic' in episode_info:
                    kinematic_count = episode_info['kinematic'].get('count', 0)
                    
                    # 检查实际文件数量是否与元数据描述一致
                    actual_count = len(self.file_paths["kinematic"].get(episode_id, []))
                    is_complete = actual_count >= kinematic_count
                    
                    self.results["file_existence_checks"].append({
                        "episode": episode_id,
                        "type": "kinematic",
                        "expected_count": kinematic_count,
                        "actual_count": actual_count,
                        "status": "pass" if is_complete else "fail",
                        "message": "运动学数据文件数量符合要求" if is_complete else f"运动学数据文件数量不足: 期望{kinematic_count}，实际{actual_count}"
                    })
                    
                    if not is_complete:
                        self.logger.warning(f"运动学数据文件数量不足: 期望{kinematic_count}，实际{actual_count}，episode={episode_id}")
                
                # 检查动力学数据文件
                if 'dynamic' in episode_info:
                    dynamic_count = episode_info['dynamic'].get('count', 0)
                    
                    # 检查实际文件数量是否与元数据描述一致
                    actual_count = len(self.file_paths["dynamic"].get(episode_id, []))
                    is_complete = actual_count >= dynamic_count
                    
                    self.results["file_existence_checks"].append({
                        "episode": episode_id,
                        "type": "dynamic",
                        "expected_count": dynamic_count,
                        "actual_count": actual_count,
                        "status": "pass" if is_complete else "fail",
                        "message": "动力学数据文件数量符合要求" if is_complete else f"动力学数据文件数量不足: 期望{dynamic_count}，实际{actual_count}"
                    })
                    
                    if not is_complete:
                        self.logger.warning(f"动力学数据文件数量不足: 期望{dynamic_count}，实际{actual_count}，episode={episode_id}")
    
    def _check_frame_index_completeness(self) -> None:
        """
        检查图像文件名和CSV文件的frame_index时间戳完整性
        """
        self.logger.info("开始检查图像文件名和CSV文件的frame_index时间戳完整性...")
        
        # 检查每个episode的图像文件名序列是否连续
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                # 提取图像文件名中的数字部分
                frame_indices = []
                for img_path in img_paths:
                    filename = os.path.basename(img_path)
                    if filename.endswith('.jpg'):
                        try:
                            frame_index = int(filename.split('.')[0])
                            frame_indices.append(frame_index)
                        except ValueError:
                            continue
                
                # 检查序列是否连续
                if frame_indices:
                    frame_indices.sort()
                    expected_indices = set(range(min(frame_indices), max(frame_indices) + 1))
                    actual_indices = set(frame_indices)
                    missing_indices = expected_indices - actual_indices
                    
                    is_complete = len(missing_indices) == 0
                    
                    self.results["frame_index_checks"].append({
                        "episode": episode,
                        "view": view,
                        "type": "image",
                        "min_index": min(frame_indices),
                        "max_index": max(frame_indices),
                        "expected_count": len(expected_indices),
                        "actual_count": len(actual_indices),
                        "missing_count": len(missing_indices),
                        "missing_indices": list(missing_indices)[:10] if len(missing_indices) > 0 else [],  # 只记录前10个缺失的索引
                        "status": "pass" if is_complete else "fail",
                        "message": "图像文件名序列完整" if is_complete else f"图像文件名序列不完整: 缺失{len(missing_indices)}个索引"
                    })
                    
                    if not is_complete:
                        self.logger.warning(f"图像文件名序列不完整: 缺失{len(missing_indices)}个索引，episode={episode}, view={view}")
        
        # 检查每个episode的CSV文件中的frame_index是否连续
        for episode, csv_paths in self.file_paths["kinematic"].items():
            for csv_path in csv_paths:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否存在frame_index列
                    if 'frame_index' in df.columns:
                        frame_indices = df['frame_index'].tolist()
                        
                        # 检查序列是否连续
                        if frame_indices:
                            frame_indices.sort()
                            expected_indices = set(range(int(min(frame_indices)), int(max(frame_indices)) + 1))
                            actual_indices = set(map(int, frame_indices))
                            missing_indices = expected_indices - actual_indices
                            
                            is_complete = len(missing_indices) == 0
                            
                            self.results["frame_index_checks"].append({
                                "file": csv_path,
                                "episode": episode,
                                "type": "kinematic",
                                "min_index": int(min(frame_indices)),
                                "max_index": int(max(frame_indices)),
                                "expected_count": len(expected_indices),
                                "actual_count": len(actual_indices),
                                "missing_count": len(missing_indices),
                                "missing_indices": list(missing_indices)[:10] if len(missing_indices) > 0 else [],  # 只记录前10个缺失的索引
                                "status": "pass" if is_complete else "fail",
                                "message": "运动学数据frame_index序列完整" if is_complete else f"运动学数据frame_index序列不完整: 缺失{len(missing_indices)}个索引"
                            })
                            
                            if not is_complete:
                                self.logger.warning(f"运动学数据frame_index序列不完整: 缺失{len(missing_indices)}个索引，file={csv_path}")
                except Exception as e:
                    self.logger.error(f"检查CSV文件frame_index时出错: {csv_path}, 错误: {str(e)}")
                    self.results["frame_index_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "type": "kinematic",
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
        
        # 检查每个episode的CSV文件中的frame_index是否连续
        for episode, csv_paths in self.file_paths["dynamic"].items():
            for csv_path in csv_paths:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否存在frame_index列
                    if 'frame_index' in df.columns:
                        frame_indices = df['frame_index'].tolist()
                        
                        # 检查序列是否连续
                        if frame_indices:
                            frame_indices.sort()
                            expected_indices = set(range(int(min(frame_indices)), int(max(frame_indices)) + 1))
                            actual_indices = set(map(int, frame_indices))
                            missing_indices = expected_indices - actual_indices
                            
                            is_complete = len(missing_indices) == 0
                            
                            self.results["frame_index_checks"].append({
                                "file": csv_path,
                                "episode": episode,
                                "type": "dynamic",
                                "min_index": int(min(frame_indices)),
                                "max_index": int(max(frame_indices)),
                                "expected_count": len(expected_indices),
                                "actual_count": len(actual_indices),
                                "missing_count": len(missing_indices),
                                "missing_indices": list(missing_indices)[:10] if len(missing_indices) > 0 else [],  # 只记录前10个缺失的索引
                                "status": "pass" if is_complete else "fail",
                                "message": "动力学数据frame_index序列完整" if is_complete else f"动力学数据frame_index序列不完整: 缺失{len(missing_indices)}个索引"
                            })
                            
                            if not is_complete:
                                self.logger.warning(f"动力学数据frame_index序列不完整: 缺失{len(missing_indices)}个索引，file={csv_path}")
                except Exception as e:
                    self.logger.error(f"检查CSV文件frame_index时出错: {csv_path}, 错误: {str(e)}")
                    self.results["frame_index_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "type": "dynamic",
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def generate_report(self, output_path: str = "completeness_check_report.json") -> None:
        """
        生成完整性检查报告
        参数:
            output_path: 输出报告的路径
        """
        # 计算统计信息
        stats = {
            "metadata": {
                "total": len(self.results["metadata_checks"]),
                "pass": sum(1 for check in self.results["metadata_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["metadata_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["metadata_checks"] if check["status"] == "error"),
                "warning": sum(1 for check in self.results["metadata_checks"] if check["status"] == "warning"),
                "pass_rate": sum(1 for check in self.results["metadata_checks"] if check["status"] == "pass") / len(self.results["metadata_checks"]) if self.results["metadata_checks"] else 0
            },
            "file_existence": {
                "total": len(self.results["file_existence_checks"]),
                "pass": sum(1 for check in self.results["file_existence_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["file_existence_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["file_existence_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["file_existence_checks"] if check["status"] == "pass") / len(self.results["file_existence_checks"]) if self.results["file_existence_checks"] else 0
            },
            "frame_index": {
                "total": len(self.results["frame_index_checks"]),
                "pass": sum(1 for check in self.results["frame_index_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["frame_index_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["frame_index_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["frame_index_checks"] if check["status"] == "pass") / len(self.results["frame_index_checks"]) if self.results["frame_index_checks"] else 0
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
        
        self.logger.info(f"完整性检查报告已保存至: {output_path}")
        
        # 打印摘要
        self.logger.info("完整性检查摘要:")
        self.logger.info(f"元数据检查: 总计 {stats['metadata']['total']}, 通过 {stats['metadata']['pass']}, 失败 {stats['metadata']['fail']}, 错误 {stats['metadata']['error']}, 警告 {stats['metadata']['warning']}")
        self.logger.info(f"文件存在性检查: 总计 {stats['file_existence']['total']}, 通过 {stats['file_existence']['pass']}, 失败 {stats['file_existence']['fail']}, 错误 {stats['file_existence']['error']}")
        self.logger.info(f"帧索引检查: 总计 {stats['frame_index']['total']}, 通过 {stats['frame_index']['pass']}, 失败 {stats['frame_index']['fail']}, 错误 {stats['frame_index']['error']}")