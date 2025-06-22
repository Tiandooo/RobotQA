import os
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from data_qa_new.qa_base import BaseChecker


class AccessibilityChecker(BaseChecker):
    """
    可访问性检查器，用于检查数据集的可访问性
    """
    def __init__(self, dataset_root: str):
        """
        初始化可访问性检查器
        参数:
            dataset_root: 数据集根目录路径
        """
        super().__init__(dataset_root)
        self.results = {
            "image_access_checks": [],
            "depth_access_checks": [],
            "video_access_checks": [],
            "csv_access_checks": []
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
        运行所有可访问性检查
        返回:
            包含所有检查结果的字典
        """
        self.logger.info(f"开始检查数据集可访问性: {self.dataset_root}")
        
        # 检查图像文件是否可访问
        self._check_image_accessibility()
        
        # 检查深度图文件是否可访问
        self._check_depth_accessibility()
        
        # 检查视频文件是否可访问
        self._check_video_accessibility()
        
        # 检查CSV文件是否可访问
        self._check_csv_accessibility()
        
        return self.results
    
    def _check_image_accessibility(self) -> None:
        """
        检查图像文件是否可访问
        """
        self.logger.info("开始检查图像文件可访问性...")
        
        # 遍历所有图像文件
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                for img_path in img_paths:
                    try:
                        # 尝试使用OpenCV读取图像
                        img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        cv_success = img_cv is not None

                        # 尝试使用PIL读取图像
                        try:
                            img_pil = Image.open(img_path)
                            pil_success = True
                        except Exception:
                            pil_success = False

                        # 记录结果
                        self.results["image_access_checks"].append({
                            "file": img_path,
                            "episode": episode,
                            "view": view,
                            "cv_success": cv_success,
                            "pil_success": pil_success,
                            "status": "pass" if (cv_success and pil_success) else "fail",
                            "message": "图像文件可正常访问" if (cv_success and pil_success) else "图像文件访问异常"
                        })

                        if not (cv_success and pil_success):
                            self.logger.warning(f"图像文件访问异常: {img_path}")
                            if not cv_success:
                                self.logger.warning(f"  - OpenCV无法读取图像")
                            if not pil_success:
                                self.logger.warning(f"  - PIL无法读取图像")
                    except Exception as e:
                        self.logger.error(f"检查图像文件可访问性时出错: {img_path}, 错误: {str(e)}")
                        self.results["image_access_checks"].append({
                            "file": img_path,
                            "episode": episode,
                            "view": view,
                            "status": "error",
                            "message": f"检查时出错: {str(e)}"
                        })
        for episode, views in self.file_paths["images"].items():
            for view, img_paths in views.items():
                # 随机抽样检查，最多检查10个文件
                sample_size = min(10, len(img_paths))
                if sample_size > 0:
                    sample_files = np.random.choice(img_paths, sample_size, replace=False)
                    
                    for img_path in sample_files:
                        try:
                            # 尝试使用OpenCV读取图像
                            img_cv = cv2.imread(img_path)
                            cv_success = img_cv is not None
                            
                            # 尝试使用PIL读取图像
                            try:
                                img_pil = Image.open(img_path)
                                pil_success = True
                            except Exception:
                                pil_success = False
                            
                            # 记录结果
                            self.results["image_access_checks"].append({
                                "file": img_path,
                                "episode": episode,
                                "view": view,
                                "cv_success": cv_success,
                                "pil_success": pil_success,
                                "status": "pass" if (cv_success and pil_success) else "fail",
                                "message": "图像文件可正常访问" if (cv_success and pil_success) else "图像文件访问异常"
                            })
                            
                            if not (cv_success and pil_success):
                                self.logger.warning(f"图像文件访问异常: {img_path}")
                                if not cv_success:
                                    self.logger.warning(f"  - OpenCV无法读取图像")
                                if not pil_success:
                                    self.logger.warning(f"  - PIL无法读取图像")
                        except Exception as e:
                            self.logger.error(f"检查图像文件可访问性时出错: {img_path}, 错误: {str(e)}")
                            self.results["image_access_checks"].append({
                                "file": img_path,
                                "episode": episode,
                                "view": view,
                                "status": "error",
                                "message": f"检查时出错: {str(e)}"
                            })
    
    def _check_depth_accessibility(self) -> None:
        """
        检查深度图文件是否可访问
        """
        self.logger.info("开始检查深度图文件可访问性...")
        
        # 遍历所有深度图文件
        for episode, img_paths in self.file_paths["depth"].items():
            for img_path in img_paths:
                try:
                    # 尝试使用OpenCV读取深度图
                    img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    cv_success = img_cv is not None
                    
                    # 尝试使用PIL读取深度图
                    try:
                        img_pil = Image.open(img_path)
                        pil_success = True
                    except Exception:
                        pil_success = False
                    
                    # 记录结果
                    self.results["depth_access_checks"].append({
                        "file": img_path,
                        "episode": episode,
                        "cv_success": cv_success,
                        "pil_success": pil_success,
                        "status": "pass" if (cv_success and pil_success) else "fail",
                        "message": "深度图文件可正常访问" if (cv_success and pil_success) else "深度图文件访问异常"
                    })
                    
                    if not (cv_success and pil_success):
                        self.logger.warning(f"深度图文件访问异常: {img_path}")
                        if not cv_success:
                            self.logger.warning(f"  - OpenCV无法读取深度图")
                        if not pil_success:
                            self.logger.warning(f"  - PIL无法读取深度图")
                except Exception as e:
                    self.logger.error(f"检查深度图文件可访问性时出错: {img_path}, 错误: {str(e)}")
                    self.results["depth_access_checks"].append({
                        "file": img_path,
                        "episode": episode,
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def _check_video_accessibility(self) -> None:
        """
        检查视频文件是否可访问
        """
        self.logger.info("开始检查视频文件可访问性...")
        
        # 遍历所有视频文件
        for episode, video_paths in self.file_paths["videos"].items():
            # 随机抽样检查，最多检查5个文件
            sample_size = min(5, len(video_paths))
            if sample_size > 0:
                sample_files = np.random.choice(video_paths, sample_size, replace=False)
                
                for video_path in sample_files:
                    try:
                        # 尝试使用OpenCV打开视频
                        cap = cv2.VideoCapture(video_path)
                        is_opened = cap.isOpened()
                        
                        # 尝试读取第一帧
                        first_frame_success = False
                        if is_opened:
                            ret, frame = cap.read()
                            first_frame_success = ret
                        
                        # 释放视频对象
                        if is_opened:
                            cap.release()
                        
                        # 记录结果
                        self.results["video_access_checks"].append({
                            "file": video_path,
                            "episode": episode,
                            "is_opened": is_opened,
                            "first_frame_success": first_frame_success,
                            "status": "pass" if (is_opened and first_frame_success) else "fail",
                            "message": "视频文件可正常访问" if (is_opened and first_frame_success) else "视频文件访问异常"
                        })
                        
                        if not (is_opened and first_frame_success):
                            self.logger.warning(f"视频文件访问异常: {video_path}")
                            if not is_opened:
                                self.logger.warning(f"  - 无法打开视频文件")
                            elif not first_frame_success:
                                self.logger.warning(f"  - 无法读取视频第一帧")
                    except Exception as e:
                        self.logger.error(f"检查视频文件可访问性时出错: {video_path}, 错误: {str(e)}")
                        self.results["video_access_checks"].append({
                            "file": video_path,
                            "episode": episode,
                            "status": "error",
                            "message": f"检查时出错: {str(e)}"
                        })
    
    def _check_csv_accessibility(self) -> None:
        """
        检查CSV文件是否可访问
        """
        self.logger.info("开始检查CSV文件可访问性...")
        
        # 合并运动学和动力学数据文件
        all_csv_files = {}
        for episode, csv_paths in self.file_paths["kinematic"].items():
            if episode not in all_csv_files:
                all_csv_files[episode] = []
            all_csv_files[episode].extend(csv_paths)
        
        for episode, csv_paths in self.file_paths["dynamic"].items():
            if episode not in all_csv_files:
                all_csv_files[episode] = []
            all_csv_files[episode].extend(csv_paths)
        
        # 遍历所有CSV文件
        for episode, csv_paths in all_csv_files.items():
            for csv_path in csv_paths:
                try:
                    # 尝试使用pandas读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否成功读取
                    read_success = df is not None and len(df) > 0
                    
                    # 记录结果
                    self.results["csv_access_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "read_success": read_success,
                        "row_count": len(df) if read_success else 0,
                        "column_count": len(df.columns) if read_success else 0,
                        "status": "pass" if read_success else "fail",
                        "message": "CSV文件可正常访问" if read_success else "CSV文件访问异常"
                    })
                    
                    if not read_success:
                        self.logger.warning(f"CSV文件访问异常: {csv_path}")
                except Exception as e:
                    self.logger.error(f"检查CSV文件可访问性时出错: {csv_path}, 错误: {str(e)}")
                    self.results["csv_access_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def generate_report(self, output_path: str = "accessibility_check_report.json") -> None:
        """
        生成可访问性检查报告
        参数:
            output_path: 输出报告的路径
        """
        # 计算统计信息
        stats = {
            "image_access": {
                "total": len(self.results["image_access_checks"]),
                "pass": sum(1 for check in self.results["image_access_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["image_access_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["image_access_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["image_access_checks"] if check["status"] == "pass") / len(self.results["image_access_checks"]) if self.results["image_access_checks"] else 0
            },
            "depth_access": {
                "total": len(self.results["depth_access_checks"]),
                "pass": sum(1 for check in self.results["depth_access_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["depth_access_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["depth_access_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["depth_access_checks"] if check["status"] == "pass") / len(self.results["depth_access_checks"]) if self.results["depth_access_checks"] else 0
            },
            "video_access": {
                "total": len(self.results["video_access_checks"]),
                "pass": sum(1 for check in self.results["video_access_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["video_access_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["video_access_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["video_access_checks"] if check["status"] == "pass") / len(self.results["video_access_checks"]) if self.results["video_access_checks"] else 0
            },
            "csv_access": {
                "total": len(self.results["csv_access_checks"]),
                "pass": sum(1 for check in self.results["csv_access_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["csv_access_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["csv_access_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["csv_access_checks"] if check["status"] == "pass") / len(self.results["csv_access_checks"]) if self.results["csv_access_checks"] else 0
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
        
        self.logger.info(f"可访问性检查报告已保存至: {output_path}")
        
        # 打印摘要
        self.logger.info("可访问性检查摘要:")
        self.logger.info(f"图像文件可访问性检查: 总计 {stats['image_access']['total']}, 通过 {stats['image_access']['pass']}, 失败 {stats['image_access']['fail']}, 错误 {stats['image_access']['error']}")
        self.logger.info(f"深度图文件可访问性检查: 总计 {stats['depth_access']['total']}, 通过 {stats['depth_access']['pass']}, 失败 {stats['depth_access']['fail']}, 错误 {stats['depth_access']['error']}")
        self.logger.info(f"视频文件可访问性检查: 总计 {stats['video_access']['total']}, 通过 {stats['video_access']['pass']}, 失败 {stats['video_access']['fail']}, 错误 {stats['video_access']['error']}")
        self.logger.info(f"CSV文件可访问性检查: 总计 {stats['csv_access']['total']}, 通过 {stats['csv_access']['pass']}, 失败 {stats['csv_access']['fail']}, 错误 {stats['csv_access']['error']}")
        

        
if __name__ == "__main__":
    dataset_root = "data/agibot_world/agibot_world_raw/356"
    print(f"开始运行可访问性检查...")
    output_dir = "reports"
    accessibility_checker = AccessibilityChecker(dataset_root)
    accessibility_results = accessibility_checker.check()
    accessibility_checker.generate_report(os.path.join(output_dir, "accessibility_check_report.json"))
