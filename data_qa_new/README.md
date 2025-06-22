# 数据集质量检查工具

这是一个用于检查机器人数据集质量的工具，包括规范性、完整性、准确性、一致性和可访问性五个方面的检查。

## 功能特点

- **规范性检查**：检查图像、视频和运动学/动力学数据的格式是否符合标准要求，包括文件命名规则和图像安全性检查。
- **完整性检查**：根据元数据描述文件检查文件是否存在，检查图像文件名和CSV文件的frame_index时间戳完整性。
- **准确性检查**：通过哈希算法检测重复图像，检查运动学/动力学数据的有效性和合理性。
- **一致性检查**：计算运动学和动力学数据中关节角度、速度、加速度的变化曲线，检测是否存在异常突变情况。
- **可访问性检查**：检查数据存储文件是否能正常打开、解析。

## 安装依赖

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 使用方法

### 命令行使用

\`\`\`bash
python -m data_qa_new.data_qa <dataset_root> [--metadata METADATA_FILE] [--output OUTPUT_DIR]
\`\`\`

参数说明：
- `dataset_root`：数据集根目录路径（必需）
- `--metadata`：元数据描述文件路径（可选，如果不指定则自动查找）
- `--output`：输出报告的目录（可选，默认为当前目录下的reports目录）

### 代码中使用

\`\`\`python
from data_qa_new.data_qa import run_all_checks

# 运行所有检查
results = run_all_checks(
    dataset_root="/path/to/dataset",
    metadata_file="/path/to/metadata.json",  # 可选
    output_dir="/path/to/output"  # 可选
)
\`\`\`

## 输出报告

工具会在指定的输出目录下生成以下报告：

- `format_check_report.json`：规范性检查报告
- `completeness_check_report.json`：完整性检查报告
- `accuracy_check_report.json`：准确性检查报告
- `consistency_check_report.json`：一致性检查报告
- `accessibility_check_report.json`：可访问性检查报告
- `combined_report.json`：综合报告，包含所有检查的摘要信息

## 数据集目录结构要求

工具假设数据集具有以下目录结构：

\`\`\`
dataset_root/
├── observations/
│   ├── episode_1/
│   │   ├── images/
│   │   │   ├── view_1/
│   │   │   │   ├── 000001.jpg
│   │   │   │   ├── 000002.jpg
│   │   │   │   └── ...
│   │   │   └── view_2/
│   │   │       ├── 000001.jpg
│   │   │       ├── 000002.jpg
│   │   │       └── ...
│   │   ├── depth/
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   └── ...
│   │   └── videos/
│   │       ├── camera1.mp4
│   │       ├── camera2.mp4
│   │       └── ...
│   └── episode_2/
│       └── ...
└── proprio_stats/
    ├── episode_1/
    │   ├── state_joint.csv
    │   ├── state_pose.csv
    │   ├── action_joint.csv
    │   └── action_torque.csv
    └── episode_2/
        └── ...
\`\`\`

## 元数据文件格式

元数据文件应为JSON格式，包含以下字段：

\`\`\`json
{
  "files": [
    {
      "path": "relative/path/to/file",
      "type": "file_type"
    }
  ],
  "episodes": [
    {
      "id": "episode_1",
      "images": [
        {
          "view": "view_1",
          "count": 100
        },
        {
          "view": "view_2",
          "count": 100
        }
      ],
      "depth": {
        "count": 100
      },
      "videos": {
        "count": 2
      },
      "kinematic": {
        "count": 2
      },
      "dynamic": {
        "count": 2
      }
    }
  ]
}
\`\`\`

## 开发者信息

如需扩展或修改检查器，可以参考以下文件：

- `qa_base.py`：基础检查器抽象类
- `format_checker.py`：规范性检查器
- `completeness_checker.py`：完整性检查器
- `accuracy_checker.py`：准确性检查器
- `consistency_checker.py`：一致性检查器
- `accessibility_checker.py`：可访问性检查器
- `data_qa.py`：主入口文件