import os
import json
from typing import Dict, List, Union


def scan_agibot_world_dataset(root_path: str) -> Dict[str, Union[Dict, List]]:
    """
    扫描精确匹配最新结构的agibot_world数据集

    参数:
        root_path: 包含observations/、proprio_stats/、task_info/的目录路径

    返回:
        完全匹配实际结构的嵌套字典
    """
    result = {
        "observations": {},
        "proprio_stats": {},
        "task_info": []
    }

    # 1. 处理observations
    obs_path = os.path.join(root_path, "observations")
    if os.path.exists(obs_path):
        for episode in os.listdir(obs_path):
            episode_path = os.path.join(obs_path, episode)
            if not os.path.isdir(episode_path):
                continue

            # 初始化episode结构
            result["observations"][episode] = {
                "images": {},
                "depth": []
            }

            # 处理多视角RGB图像 (images/下的子目录)
            images_path = os.path.join(episode_path, "images")
            if os.path.exists(images_path):
                for view_dir in os.listdir(images_path):
                    view_path = os.path.join(images_path, view_dir)
                    if os.path.isdir(view_path):
                        result["observations"][episode]["images"][view_dir] = [
                            f for f in os.listdir(view_path)
                            if f.endswith('.jpg')
                        ]

            # 处理深度图 (depth/目录下的png文件)
            depth_path = os.path.join(episode_path, "depth")
            if os.path.exists(depth_path):
                result["observations"][episode]["depth"] = [
                    f for f in os.listdir(depth_path)
                    if f.endswith('.png')
                ]

    # 2. 处理proprio_stats
    prop_path = os.path.join(root_path, "proprio_stats")
    if os.path.exists(prop_path):
        for episode in os.listdir(prop_path):
            episode_path = os.path.join(prop_path, episode)
            if not os.path.isdir(episode_path):
                continue

            result["proprio_stats"][episode] = [
                f for f in os.listdir(episode_path)
                if f.endswith('.csv') and f.startswith('action_')
            ]

    # 3. 处理task_info
    task_path = os.path.join(root_path, "task_info")
    if os.path.exists(task_path):
        result["task_info"] = [
            f for f in os.listdir(task_path)
            if f.startswith('task_') and f.endswith('.json')
        ]

    return result


def get_dataset_structure_json(root_path: str) -> str:
    """
    生成完全匹配实际结构的JSON

    参数:
        root_path: 数据集根目录（包含observations/等一级目录）

    返回:
        格式化后的JSON字符串，包含完整嵌套结构
    """
    structure = scan_agibot_world_dataset(root_path)
    return json.dumps(structure, indent=2, ensure_ascii=False)

filenames = get_dataset_structure_json("data/agibot_world/agibot_world_raw/356")
print(filenames)