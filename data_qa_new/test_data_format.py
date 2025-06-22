#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据格式检查器
"""

import os
import sys
import argparse
from data_qa_new.old.data_format_checker import DataFormatChecker

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试数据格式检查器')
    parser.add_argument('dataset_root', nargs='?', default="data/agibot_world/agibot_world_raw/356",
                        help='数据集根目录路径')
    parser.add_argument('--output', type=str, default="data_format_check_report.json",
                        help='输出报告路径')
    parser.add_argument('--naming-rules', type=str, help='命名规则配置文件路径（JSON格式）')

    args = parser.parse_args()
    # 检查目录是否存在
    if not os.path.exists(args.dataset_root):
        print(f"错误: 数据集目录不存在: {args.dataset_root}")
        return 1

    print(f"开始检查数据集: {args.dataset_root}")
    # 创建检查器
    checker = DataFormatChecker(args.dataset_root)

    # 如果提供了命名规则配置文件，则加载自定义规则
    if args.naming_rules and os.path.exists(args.naming_rules):
        try:
            import json
            with open(args.naming_rules, 'r', encoding='utf-8') as f:
                custom_rules = json.load(f)
            checker.set_naming_rules(custom_rules)
            print(f"已加载自定义命名规则: {args.naming_rules}")
        except Exception as e:
            print(f"加载命名规则配置文件时出错: {str(e)}")
    # 运行所有检查
    checker.check_all()
    # 生成报告
    checker.generate_report(args.output)

    print(f"检查完成，报告已保存至 {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())