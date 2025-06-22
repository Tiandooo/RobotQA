# 数据格式检查器

这个工具用于检查数据集中的图像、视频和运动学/动力学数据的格式是否符合标准要求。

## 功能

1. **图像检查**：
   - 检查图像像素数据类型是否为uint8
   - 提取图像分辨率、通道数等信息
   - 支持JPEG和PNG格式

2. **视频检查**：
   - 检查视频帧数据类型是否为uint8
   - 检查视频帧数据类型是否一致
   - 提取视频分辨率、帧率等信息
   - 支持MP4格式

3. **运动学数据检查**：
   - 检查数值是否为float64类型
   - 识别不符合要求的数据列

4. **动力学数据检查**：
   - 检查数值是否为float64类型
   - 识别不符合要求的数据列

## 安装依赖

确保已安装以下依赖：

\`\`\`bash
pip install opencv-python numpy pandas
\`\`\`

## 使用方法

### 命令行使用

\`\`\`bash
python utils/data_format_checker.py <数据集根目录> [--output <输出报告路径>]
\`\`\`

例如：

\`\`\`bash
python utils/data_format_checker.py data/agibot_world/agibot_world_raw/356
\`\`\`

或者使用测试脚本：

\`\`\`bash
python test_data_format.py [数据集根目录]
\`\`\`

### 在Python代码中使用

\`\`\`python
from utils.data_format_checker import DataFormatChecker

# 创建检查器
checker = DataFormatChecker("data/agibot_world/agibot_world_raw/356")

# 运行所有检查
results = checker.check_all()

# 生成报告
checker.generate_report("data_format_check_report.json")
\`\`\`

## 输出报告

检查器会生成一个JSON格式的报告，包含以下内容：

1. 数据集路径
2. 检查时间戳
3. 统计信息（通过/失败/错误数量）
4. 详细检查结果

## 数据集结构要求

检查器假设数据集具有以下结构：

\`\`\`
dataset_root/
├── observations/
│   └── <episode_id>/
│       ├── images/
│       │   └── <view_name>/
│       │       └── *.jpg
│       ├── depth/
│       │   └── *.png
│       └── videos/
│           └── *.mp4 (可选)
└── proprio_stats/
    └── <episode_id>/
        ├── state_*.csv
        ├── action_*.csv
        └── force_*.csv / torque_*.csv (可选)
\`\`\`

## 标准要求

1. 图像像素数据类型：uint8（8位无符号整数，范围0-255）
2. 视频帧数据类型：uint8（8位无符号整数，范围0-255）
3. 运动学/动力学数据类型：float64（双精度浮点数，IEEE 754标准）
