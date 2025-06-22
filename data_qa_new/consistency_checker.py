import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from data_qa_new.qa_base import BaseChecker

class ConsistencyChecker(BaseChecker):
    """
    一致性检查器，用于检查数据集的一致性
    """
    def __init__(self, dataset_root: str):
        """
        初始化一致性检查器
        参数:
            dataset_root: 数据集根目录路径
        """
        super().__init__(dataset_root)
        self.results = {
            "kinematic_curve_checks": [],
            "dynamic_curve_checks": []
        }
        
        # 存储文件路径的数据结构
        self.file_paths = {
            "kinematic": {},  # 格式: {episode: [file_paths]}
            "dynamic": {}   # 格式: {episode: [file_paths]}
        }
        
        # 异常变化阈值
        self.anomaly_thresholds = {
            # 关节角度变化阈值 (弧度/帧)
            "joint_angle": 0.5,
            # 关节速度变化阈值 (弧度/秒/帧)
            "joint_velocity": 1.0,
            # 关节加速度变化阈值 (弧度/秒^2/帧)
            "joint_acceleration": 2.0,
            # 位置变化阈值 (米/帧)
            "position": 0.2,
            # 速度变化阈值 (米/秒/帧)
            "velocity": 0.5,
            # 加速度变化阈值 (米/秒^2/帧)
            "acceleration": 1.0,
            # 力变化阈值 (牛顿/帧)
            "force": 20.0,
            # 扭矩变化阈值 (牛米/帧)
            "torque": 10.0,
            # 电流变化阈值 (安培/帧)
            "current": 5.0,
            # 电压变化阈值 (伏特/帧)
            "voltage": 10.0
        }
        
        # 扫描目录结构
        self._scan_directory_structure()
        
    def _scan_directory_structure(self) -> None:
        """
        扫描目录结构，将文件路径保存到数据结构中
        """
        self.logger.info(f"扫描数据集目录结构: {self.dataset_root}")
        
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
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['kinematic'])} 个运动学数据目录")
        self.logger.info(f"扫描完成，找到 {len(self.file_paths['dynamic'])} 个动力学数据目录")
    
    def check(self) -> Dict[str, Any]:
        """
        运行所有一致性检查
        返回:
            包含所有检查结果的字典
        """
        self.logger.info(f"开始检查数据集一致性: {self.dataset_root}")
        
        # 检查运动学数据的变化曲线
        self._check_kinematic_curves()
        
        # 检查动力学数据的变化曲线
        self._check_dynamic_curves()
        
        return self.results
    
    def _detect_anomalies(self, data: np.ndarray, threshold: float) -> List[int]:
        """
        检测数据中的异常变化点
        参数:
            data: 数据数组
            threshold: 异常变化阈值
        返回:
            异常变化点的索引列表
        """
        # 计算一阶差分
        diff = np.abs(np.diff(data))
        
        # 找出超过阈值的点
        anomaly_indices = np.where(diff > threshold)[0]
        
        return anomaly_indices.tolist()
    
    def _check_kinematic_curves(self) -> None:
        """
        检查运动学数据的变化曲线
        """
        self.logger.info("开始检查运动学数据的变化曲线...")
        
        # 遍历所有运动学数据文件
        for episode, csv_paths in self.file_paths["kinematic"].items():
            for csv_path in csv_paths:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否存在frame_index列
                    if 'frame_index' not in df.columns:
                        self.logger.warning(f"CSV文件缺少frame_index列: {csv_path}")
                        continue
                    
                    # 按frame_index排序
                    df = df.sort_values('frame_index')
                    
                    # 检查每列数据的变化曲线
                    anomalies = {}
                    for col in df.columns:
                        if col == 'frame_index':
                            continue
                        
                        # 根据列名判断数据类型
                        threshold_key = None
                        for key in self.anomaly_thresholds.keys():
                            if key in col.lower():
                                threshold_key = key
                                break
                        
                        if threshold_key:
                            # 获取数据和阈值
                            data = df[col].values
                            threshold = self.anomaly_thresholds[threshold_key]
                            
                            # 检测异常变化点
                            anomaly_indices = self._detect_anomalies(data, threshold)
                            
                            if anomaly_indices:
                                # 记录异常变化点
                                anomalies[col] = {
                                    "threshold": threshold,
                                    "anomaly_indices": anomaly_indices,
                                    "anomaly_values": [float(data[i+1] - data[i]) for i in anomaly_indices],
                                    "anomaly_frames": [int(df.iloc[i]['frame_index']) for i in anomaly_indices]
                                }
                    
                    # 记录结果
                    self.results["kinematic_curve_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "anomaly_columns": len(anomalies),
                        "anomalies": anomalies,
                        "status": "pass" if not anomalies else "fail",
                        "message": "运动学数据变化曲线正常" if not anomalies else f"运动学数据变化曲线存在{len(anomalies)}列异常"
                    })
                    
                    if anomalies:
                        self.logger.warning(f"运动学数据变化曲线存在异常: {csv_path}")
                        for col, anomaly_info in anomalies.items():
                            self.logger.warning(f"  - 列 {col} 存在 {len(anomaly_info['anomaly_indices'])} 个异常变化点")
                except Exception as e:
                    self.logger.error(f"检查运动学数据变化曲线时出错: {csv_path}, 错误: {str(e)}")
                    self.results["kinematic_curve_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def _check_dynamic_curves(self) -> None:
        """
        检查动力学数据的变化曲线
        """
        self.logger.info("开始检查动力学数据的变化曲线...")
        
        # 遍历所有动力学数据文件
        for episode, csv_paths in self.file_paths["dynamic"].items():
            for csv_path in csv_paths:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否存在frame_index列
                    if 'frame_index' not in df.columns:
                        self.logger.warning(f"CSV文件缺少frame_index列: {csv_path}")
                        continue
                    
                    # 按frame_index排序
                    df = df.sort_values('frame_index')
                    
                    # 检查每列数据的变化曲线
                    anomalies = {}
                    for col in df.columns:
                        if col == 'frame_index':
                            continue
                        
                        # 根据列名判断数据类型
                        threshold_key = None
                        for key in self.anomaly_thresholds.keys():
                            if key in col.lower():
                                threshold_key = key
                                break
                        
                        if threshold_key:
                            # 获取数据和阈值
                            data = df[col].values
                            threshold = self.anomaly_thresholds[threshold_key]
                            
                            # 检测异常变化点
                            anomaly_indices = self._detect_anomalies(data, threshold)
                            
                            if anomaly_indices:
                                # 记录异常变化点
                                anomalies[col] = {
                                    "threshold": threshold,
                                    "anomaly_indices": anomaly_indices,
                                    "anomaly_values": [float(data[i+1] - data[i]) for i in anomaly_indices],
                                    "anomaly_frames": [int(df.iloc[i]['frame_index']) for i in anomaly_indices]
                                }
                    
                    # 记录结果
                    self.results["dynamic_curve_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "anomaly_columns": len(anomalies),
                        "anomalies": anomalies,
                        "status": "pass" if not anomalies else "fail",
                        "message": "动力学数据变化曲线正常" if not anomalies else f"动力学数据变化曲线存在{len(anomalies)}列异常"
                    })
                    
                    if anomalies:
                        self.logger.warning(f"动力学数据变化曲线存在异常: {csv_path}")
                        for col, anomaly_info in anomalies.items():
                            self.logger.warning(f"  - 列 {col} 存在 {len(anomaly_info['anomaly_indices'])} 个异常变化点")
                except Exception as e:
                    self.logger.error(f"检查动力学数据变化曲线时出错: {csv_path}, 错误: {str(e)}")
                    self.results["dynamic_curve_checks"].append({
                        "file": csv_path,
                        "episode": episode,
                        "status": "error",
                        "message": f"检查时出错: {str(e)}"
                    })
    
    def generate_report(self, output_path: str = "consistency_check_report.json") -> None:
        """
        生成一致性检查报告
        参数:
            output_path: 输出报告的路径
        """
        # 计算统计信息
        stats = {
            "kinematic_curve": {
                "total": len(self.results["kinematic_curve_checks"]),
                "pass": sum(1 for check in self.results["kinematic_curve_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["kinematic_curve_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["kinematic_curve_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["kinematic_curve_checks"] if check["status"] == "pass") / len(self.results["kinematic_curve_checks"]) if self.results["kinematic_curve_checks"] else 0
            },
            "dynamic_curve": {
                "total": len(self.results["dynamic_curve_checks"]),
                "pass": sum(1 for check in self.results["dynamic_curve_checks"] if check["status"] == "pass"),
                "fail": sum(1 for check in self.results["dynamic_curve_checks"] if check["status"] == "fail"),
                "error": sum(1 for check in self.results["dynamic_curve_checks"] if check["status"] == "error"),
                "pass_rate": sum(1 for check in self.results["dynamic_curve_checks"] if check["status"] == "pass") / len(self.results["dynamic_curve_checks"]) if self.results["dynamic_curve_checks"] else 0
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
        
        self.logger.info(f"一致性检查报告已保存至: {output_path}")
        
        # 打印摘要
        self.logger.info("一致性检查摘要:")
        self.logger.info(f"运动学数据变化曲线检查: 总计 {stats['kinematic_curve']['total']}, 通过 {stats['kinematic_curve']['pass']}, 失败 {stats['kinematic_curve']['fail']}, 错误 {stats['kinematic_curve']['error']}")
        self.logger.info(f"动力学数据变化曲线检查: 总计 {stats['dynamic_curve']['total']}, 通过 {stats['dynamic_curve']['pass']}, 失败 {stats['dynamic_curve']['fail']}, 错误 {stats['dynamic_curve']['error']}")


if __name__ == "__main__":
    dataset_root = "data/agibot_world/agibot_world_raw/356"
    output_dir = "reports"
    # 运行一致性检查
    print(f"开始运行一致性检查...")
    consistency_checker = ConsistencyChecker(dataset_root)
    consistency_results = consistency_checker.check()
    consistency_checker.generate_report(os.path.join(output_dir, "consistency_check_report.json"))
