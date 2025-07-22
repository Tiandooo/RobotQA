# 数据集质量检查API

这是一个基于FastAPI的数据集质量检查API服务，提供启动评估、查询状态和关闭任务三个接口。

## 功能特点

- **启动评估**：启动一个新的数据集质量评估任务
- **查询状态**：查询指定任务的状态和进度
- **关闭任务**：关闭指定的任务

## 部署方法

### 使用Docker部署

1. 构建Docker镜像

\`\`\`bash
cd data_qa_new
docker build -t data-qa-api .
\`\`\`

2. 运行Docker容器

\`\`\`bash
docker run -d -p 8000:8000 -v /path/to/datasets:/datasets --name data-qa-api-container data-qa-api
\`\`\`

注意：需要将本地的数据集目录挂载到容器中，以便API服务能够访问数据集。

### 直接运行

1. 安装依赖

\`\`\`bash
pip install -r requirements.txt
pip install fastapi uvicorn
\`\`\`

2. 运行API服务

\`\`\`bash
cd data_qa_new
uvicorn api:app --host 0.0.0.0 --port 8000
\`\`\`

## API接口说明

### 1. 启动评估任务

- **URL**: `/tasks`
- **方法**: `POST`
- **请求体**:
  \`\`\`json
  {
    "dataset_root": "/datasets/my_dataset",
    "metadata_file": "/datasets/my_dataset/metadata.json",
    "output_dir": "/app/reports/my_task"
  }
  \`\`\`
  - `dataset_root`: 数据集根目录路径（必需）
  - `metadata_file`: 元数据描述文件路径（可选）
  - `output_dir`: 输出报告的目录（可选）

- **响应**:
  \`\`\`json
  {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending",
    "message": "任务已创建，等待执行",
    "created_at": 1616161616.0,
    "dataset_root": "/datasets/my_dataset",
    "output_dir": "/app/reports/my_task"
  }
  \`\`\`

### 2. 查询任务状态

- **URL**: `/tasks/{task_id}`
- **方法**: `GET`
- **响应**:
  \`\`\`json
  {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "message": "任务正在运行中",
    "progress": 50.0
  }
  \`\`\`
  或者（当任务完成时）:
  \`\`\`json
  {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "message": "任务已完成",
    "progress": 100.0,
    "result_path": "/app/reports/my_task/combined_report.json"
  }
  \`\`\`

### 3. 关闭任务

- **URL**: `/tasks/{task_id}`
- **方法**: `DELETE`
- **响应**:
  \`\`\`json
  {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "cancelled",
    "message": "任务已取消",
    "progress": 50.0
  }
  \`\`\`

## 使用示例

### 使用curl

1. 启动评估任务

```bash
curl -X POST "http://localhost:8000/tasks" \
     -H "Content-Type: application/json" \
     -d '{"dataset_root": "/home/server/fdu/RobotQA/data/agibot_world/agibot_world_raw/356"}'
```

2. 查询任务状态

\`\`\`bash
curl -X GET "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000"
\`\`\`

3. 关闭任务

\`\`\`bash
curl -X DELETE "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000"
\`\`\`

### 使用Python

\`\`\`python
import requests
import json

# 启动评估任务
response = requests.post(
    "http://localhost:8000/tasks",
    json={"dataset_root": "/datasets/my_dataset"}
)
task = response.json()
task_id = task["task_id"]
print(f"任务ID: {task_id}")

# 查询任务状态
response = requests.get(f"http://localhost:8000/tasks/{task_id}")
status = response.json()
print(f"任务状态: {status['status']}")

# 关闭任务
response = requests.delete(f"http://localhost:8000/tasks/{task_id}")
result = response.json()
print(f"任务已关闭: {result['status']}")
\`\`\`

## API文档

启动API服务后，可以通过访问以下URL查看自动生成的API文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`