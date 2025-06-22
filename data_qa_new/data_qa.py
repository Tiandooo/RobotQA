import argparse
import json
import os
import time
from typing import Dict, Any, List

import pandas as pd

from data_qa_new.format_checker import FormatChecker
from data_qa_new.completeness_checker import CompletenessChecker
from data_qa_new.accuracy_checker import AccuracyChecker
from data_qa_new.consistency_checker import ConsistencyChecker
from data_qa_new.accessibility_checker import AccessibilityChecker

def run_all_checks(dataset_root: str, metadata_file: str = None, output_dir: str = None) -> Dict[str, Any]:
    """
    运行所有检查
    参数:
        dataset_root: 数据集根目录路径
        metadata_file: 元数据描述文件路径，如果为None则自动查找
        output_dir: 输出报告的目录，如果为None则使用当前目录
    返回:
        包含所有检查结果的字典
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "reports")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行规范性检查
    print(f"开始运行规范性检查...")
    format_checker = FormatChecker(dataset_root)
    format_results = format_checker.check()
    format_checker.generate_report(os.path.join(output_dir, "format_check_report.json"))
    
    # 运行完整性检查
    # print(f"开始运行完整性检查...")
    # completeness_checker = CompletenessChecker(dataset_root, metadata_file)
    # completeness_results = completeness_checker.check()
    # completeness_checker.generate_report(os.path.join(output_dir, "completeness_check_report.json"))
    
    # 运行准确性检查
    print(f"开始运行准确性检查...")
    accuracy_checker = AccuracyChecker(dataset_root)
    accuracy_results = accuracy_checker.check()
    accuracy_checker.generate_report(os.path.join(output_dir, "accuracy_check_report.json"))
    
    # 运行一致性检查
    # print(f"开始运行一致性检查...")
    # consistency_checker = ConsistencyChecker(dataset_root)
    # consistency_results = consistency_checker.check()
    # consistency_checker.generate_report(os.path.join(output_dir, "consistency_check_report.json"))
    #
    # 运行可访问性检查
    print(f"开始运行可访问性检查...")
    accessibility_checker = AccessibilityChecker(dataset_root)
    accessibility_results = accessibility_checker.check()
    accessibility_checker.generate_report(os.path.join(output_dir, "accessibility_check_report.json"))
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 生成综合报告
    combined_report = {
        "dataset_root": dataset_root,
        "timestamp": pd.Timestamp.now().isoformat(),
        "elapsed_time": elapsed_time,
        "summary": {
            "format": {
                "image_checks": {
                    "total": len(format_results["image_checks"]),
                    "pass": sum(1 for check in format_results["image_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in format_results["image_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in format_results["image_checks"] if check["status"] == "error")
                },
                "video_checks": {
                    "total": len(format_results["video_checks"]),
                    "pass": sum(1 for check in format_results["video_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in format_results["video_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in format_results["video_checks"] if check["status"] == "error")
                },
                "kinematic_checks": {
                    "total": len(format_results["kinematic_checks"]),
                    "pass": sum(1 for check in format_results["kinematic_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in format_results["kinematic_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in format_results["kinematic_checks"] if check["status"] == "error")
                },
                "dynamic_checks": {
                    "total": len(format_results["dynamic_checks"]),
                    "pass": sum(1 for check in format_results["dynamic_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in format_results["dynamic_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in format_results["dynamic_checks"] if check["status"] == "error")
                },
                "naming_checks": {
                    "total": len(format_results["naming_checks"]),
                    "pass": sum(1 for check in format_results["naming_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in format_results["naming_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in format_results["naming_checks"] if check["status"] == "error")
                },
                "safety_checks": {
                    "total": len(format_results["safety_checks"]),
                    "pass": sum(1 for check in format_results["safety_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in format_results["safety_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in format_results["safety_checks"] if check["status"] == "error")
                }
            },
            # "completeness": {
            #     "metadata_checks": {
            #         "total": len(completeness_results["metadata_checks"]),
            #         "pass": sum(1 for check in completeness_results["metadata_checks"] if check["status"] == "pass"),
            #         "fail": sum(1 for check in completeness_results["metadata_checks"] if check["status"] == "fail"),
            #         "error": sum(1 for check in completeness_results["metadata_checks"] if check["status"] == "error"),
            #         "warning": sum(1 for check in completeness_results["metadata_checks"] if check["status"] == "warning")
            #     },
            #     "file_existence_checks": {
            #         "total": len(completeness_results["file_existence_checks"]),
            #         "pass": sum(1 for check in completeness_results["file_existence_checks"] if check["status"] == "pass"),
            #         "fail": sum(1 for check in completeness_results["file_existence_checks"] if check["status"] == "fail"),
            #         "error": sum(1 for check in completeness_results["file_existence_checks"] if check["status"] == "error")
            #     },
            #     "frame_index_checks": {
            #         "total": len(completeness_results["frame_index_checks"]),
            #         "pass": sum(1 for check in completeness_results["frame_index_checks"] if check["status"] == "pass"),
            #         "fail": sum(1 for check in completeness_results["frame_index_checks"] if check["status"] == "fail"),
            #         "error": sum(1 for check in completeness_results["frame_index_checks"] if check["status"] == "error")
            #     }
            # },
            "accuracy": {
                "image_hash_checks": {
                    "total": len(accuracy_results["image_hash_checks"]),
                    "pass": 0,  # 哈希检查没有pass状态，只有fail状态
                    "fail": len(accuracy_results["image_hash_checks"]),
                    "error": 0
                },
                "kinematic_value_checks": {
                    "total": len(accuracy_results["kinematic_value_checks"]),
                    "pass": sum(1 for check in accuracy_results["kinematic_value_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in accuracy_results["kinematic_value_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in accuracy_results["kinematic_value_checks"] if check["status"] == "error")
                },
                "dynamic_value_checks": {
                    "total": len(accuracy_results["dynamic_value_checks"]),
                    "pass": sum(1 for check in accuracy_results["dynamic_value_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in accuracy_results["dynamic_value_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in accuracy_results["dynamic_value_checks"] if check["status"] == "error")
                }
            },
            # "consistency": {
            #     "kinematic_curve_checks": {
            #         "total": len(consistency_results["kinematic_curve_checks"]),
            #         "pass": sum(1 for check in consistency_results["kinematic_curve_checks"] if check["status"] == "pass"),
            #         "fail": sum(1 for check in consistency_results["kinematic_curve_checks"] if check["status"] == "fail"),
            #         "error": sum(1 for check in consistency_results["kinematic_curve_checks"] if check["status"] == "error")
            #     },
            #     "dynamic_curve_checks": {
            #         "total": len(consistency_results["dynamic_curve_checks"]),
            #         "pass": sum(1 for check in consistency_results["dynamic_curve_checks"] if check["status"] == "pass"),
            #         "fail": sum(1 for check in consistency_results["dynamic_curve_checks"] if check["status"] == "fail"),
            #         "error": sum(1 for check in consistency_results["dynamic_curve_checks"] if check["status"] == "error")
            #     }
            # },
            "accessibility": {
                "image_access_checks": {
                    "total": len(accessibility_results["image_access_checks"]),
                    "pass": sum(1 for check in accessibility_results["image_access_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in accessibility_results["image_access_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in accessibility_results["image_access_checks"] if check["status"] == "error")
                },
                "depth_access_checks": {
                    "total": len(accessibility_results["depth_access_checks"]),
                    "pass": sum(1 for check in accessibility_results["depth_access_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in accessibility_results["depth_access_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in accessibility_results["depth_access_checks"] if check["status"] == "error")
                },
                "video_access_checks": {
                    "total": len(accessibility_results["video_access_checks"]),
                    "pass": sum(1 for check in accessibility_results["video_access_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in accessibility_results["video_access_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in accessibility_results["video_access_checks"] if check["status"] == "error")
                },
                "csv_access_checks": {
                    "total": len(accessibility_results["csv_access_checks"]),
                    "pass": sum(1 for check in accessibility_results["csv_access_checks"] if check["status"] == "pass"),
                    "fail": sum(1 for check in accessibility_results["csv_access_checks"] if check["status"] == "fail"),
                    "error": sum(1 for check in accessibility_results["csv_access_checks"] if check["status"] == "error")
                }
            }
        }
    }
    
    # 保存综合报告
    with open(os.path.join(output_dir, "combined_report.json"), 'w', encoding='utf-8') as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)
    
    print(f"所有检查完成，耗时 {elapsed_time:.2f} 秒")
    print(f"报告已保存至 {output_dir}")
    
    return combined_report

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='数据集质量检查工具')
    parser.add_argument('dataset_root', type=str, help='数据集根目录路径')
    parser.add_argument('--metadata', type=str, default=None, help='元数据描述文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出报告的目录')

    args = parser.parse_args()

    # 运行所有检查
    run_all_checks(args.dataset_root, args.metadata, args.output)

if __name__ == "__main__":
    main()