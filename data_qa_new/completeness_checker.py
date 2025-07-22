import json
import os
from typing import Dict, Any

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

        # 检查元数据中的episodes列表
        if self.metadata and isinstance(self.metadata, list):
            for episode_info in self.metadata:
                episode_id = str(episode_info.get('episode_id', ''))

                if not episode_id:
                    self.logger.warning("发现没有episode_id的记录")
                    continue

                # 检查observations目录下是否存在该episode
                obs_episode_path = os.path.join(self.dataset_root, "observations", episode_id)
                obs_exists = os.path.exists(obs_episode_path) and os.path.isdir(obs_episode_path)

                self.results["file_existence_checks"].append({
                    "episode": episode_id,
                    "type": "observations_directory",
                    "path": obs_episode_path,
                    "status": "pass" if obs_exists else "fail",
                    "message": f"observations目录中episode {episode_id}存在" if obs_exists else f"observations目录中episode {episode_id}不存在"
                })

                if not obs_exists:
                    self.logger.warning(f"observations目录中episode {episode_id}不存在: {obs_episode_path}")

                # 检查proprio_stats目录下是否存在该episode
                proprio_episode_path = os.path.join(self.dataset_root, "proprio_stats", episode_id)
                proprio_exists = os.path.exists(proprio_episode_path) and os.path.isdir(proprio_episode_path)

                self.results["file_existence_checks"].append({
                    "episode": episode_id,
                    "type": "proprio_stats_directory",
                    "path": proprio_episode_path,
                    "status": "pass" if proprio_exists else "fail",
                    "message": f"proprio_stats目录中episode {episode_id}存在" if proprio_exists else f"proprio_stats目录中episode {episode_id}不存在"
                })

                if not proprio_exists:
                    self.logger.warning(f"proprio_stats目录中episode {episode_id}不存在: {proprio_episode_path}")

                # 如果两个目录都存在，进一步检查文件完整性
                if obs_exists and proprio_exists:
                    self._check_episode_files(episode_id, obs_episode_path, proprio_episode_path)

        # 如果metadata不是预期的格式，记录警告
        elif self.metadata:
            self.logger.warning("元数据格式不符合预期，应该是包含episode信息的数组")
            self.results["file_existence_checks"].append({
                "status": "warning",
                "message": "元数据格式不符合预期，应该是包含episode信息的数组"
            })

    def _check_episode_files(self, episode_id: str, obs_path: str, proprio_path: str) -> None:
        """
        检查单个episode的文件完整性
        """
        # 检查observations目录下的文件结构
        expected_obs_dirs = ["images", "depth", "videos"]
        for dir_name in expected_obs_dirs:
            dir_path = os.path.join(obs_path, dir_name)
            dir_exists = os.path.exists(dir_path) and os.path.isdir(dir_path)

            self.results["file_existence_checks"].append({
                "episode": episode_id,
                "type": f"observations_{dir_name}_directory",
                "path": dir_path,
                "status": "pass" if dir_exists else "fail",
                "message": f"episode {episode_id}的{dir_name}目录存在" if dir_exists else f"episode {episode_id}的{dir_name}目录不存在"
            })

            if not dir_exists:
                self.logger.warning(f"episode {episode_id}的{dir_name}目录不存在: {dir_path}")

        # 检查proprio_stats目录下的CSV文件
        csv_files = [f for f in os.listdir(proprio_path) if f.endswith('.csv')]
        kinematic_files = [f for f in csv_files if f.startswith('state_')]
        dynamic_files = [f for f in csv_files if f.startswith('action_')]

        self.results["file_existence_checks"].append({
            "episode": episode_id,
            "type": "kinematic_files",
            "path": proprio_path,
            "count": len(kinematic_files),
            "files": kinematic_files,
            "status": "pass" if kinematic_files else "fail",
            "message": f"episode {episode_id}包含{len(kinematic_files)}个运动学文件" if kinematic_files else f"episode {episode_id}缺少运动学文件"
        })

        self.results["file_existence_checks"].append({
            "episode": episode_id,
            "type": "dynamic_files",
            "path": proprio_path,
            "count": len(dynamic_files),
            "files": dynamic_files,
            "status": "pass" if dynamic_files else "fail",
            "message": f"episode {episode_id}包含{len(dynamic_files)}个动力学文件" if dynamic_files else f"episode {episode_id}缺少动力学文件"
        })

        if not kinematic_files:
            self.logger.warning(f"episode {episode_id}缺少运动学文件")
        if not dynamic_files:
            self.logger.warning(f"episode {episode_id}缺少动力学文件")

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


if __name__ == "__main__":
    dataset_root = "data/agibot_world/agibot_world_raw/356"
    metadata = "data/agibot_world/agibot_world_raw/356/task_info/info.json"
    output_dir = "reports"
    # 运行一致性检查
    print(f"开始运行一致性检查...")
    completeness_checker = CompletenessChecker(dataset_root, metadata)
    completeness_results = completeness_checker.check()
    completeness_checker.generate_report(os.path.join(output_dir, "completeness_check_report.json"))
