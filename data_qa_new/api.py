import os
import uuid
import json
import threading
import time
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from data_qa_new.data_qa import run_all_checks

app = FastAPI(title="数据集质量检查API", description="提供数据集质量检查服务的API")

# 任务状态枚举
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 任务请求模型
class TaskRequest(BaseModel):
    dataset_root: str = Field(..., description="数据集根目录路径")
    metadata_file: Optional[str] = Field(None, description="元数据描述文件路径，如果不指定则自动查找")
    output_dir: Optional[str] = Field(None, description="输出报告的目录，如果不指定则使用默认目录")

# 任务响应模型
class TaskResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="任务消息")
    created_at: float = Field(..., description="任务创建时间戳")
    started_at: Optional[float] = Field(None, description="任务开始时间戳")
    completed_at: Optional[float] = Field(None, description="任务完成时间戳")
    dataset_root: str = Field(..., description="数据集根目录路径")
    output_dir: Optional[str] = Field(None, description="输出报告的目录")
    progress: Optional[float] = Field(None, description="任务进度，0-100")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")

# 任务状态查询响应模型
class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="任务消息")
    progress: Optional[float] = Field(None, description="任务进度，0-100")
    result_path: Optional[str] = Field(None, description="结果报告路径")

# 存储所有任务的字典
tasks: Dict[str, TaskResponse] = {}

# 任务锁，防止并发问题
task_lock = threading.Lock()

def run_task(task_id: str, dataset_root: str, metadata_file: Optional[str] = None, output_dir: Optional[str] = None):
    """
    在后台运行任务
    """
    try:
        # 更新任务状态为运行中
        with task_lock:
            if task_id not in tasks:
                return
            
            tasks[task_id].status = TaskStatus.RUNNING
            tasks[task_id].started_at = time.time()
            tasks[task_id].message = "任务正在运行中"
            tasks[task_id].progress = 0.0
        
        # 设置默认输出目录
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "reports", task_id)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 更新输出目录
        with task_lock:
            tasks[task_id].output_dir = output_dir
        
        # 运行检查
        result = run_all_checks(dataset_root, metadata_file, output_dir)
        
        # 更新任务状态为已完成
        with task_lock:
            if task_id not in tasks:
                return
            
            tasks[task_id].status = TaskStatus.COMPLETED
            tasks[task_id].completed_at = time.time()
            tasks[task_id].message = "任务已完成"
            tasks[task_id].progress = 100.0
            tasks[task_id].result = result
    
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id not in tasks:
                return
            
            tasks[task_id].status = TaskStatus.FAILED
            tasks[task_id].completed_at = time.time()
            tasks[task_id].message = f"任务失败: {str(e)}"
            tasks[task_id].progress = 0.0

@app.post("/tasks", response_model=TaskResponse, summary="启动评估任务", description="启动一个新的数据集质量评估任务")
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """
    启动一个新的数据集质量评估任务
    """
    # 检查数据集根目录是否存在
    if not os.path.exists(task_request.dataset_root):
        raise HTTPException(status_code=400, detail=f"数据集根目录不存在: {task_request.dataset_root}")
    
    # 检查元数据文件是否存在（如果指定了）
    if task_request.metadata_file and not os.path.exists(task_request.metadata_file):
        raise HTTPException(status_code=400, detail=f"元数据文件不存在: {task_request.metadata_file}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务
    task = TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="任务已创建，等待执行",
        created_at=time.time(),
        dataset_root=task_request.dataset_root,
        output_dir=task_request.output_dir
    )
    
    # 存储任务
    with task_lock:
        tasks[task_id] = task
    
    # 在后台运行任务
    background_tasks.add_task(
        run_task,
        task_id=task_id,
        dataset_root=task_request.dataset_root,
        metadata_file=task_request.metadata_file,
        output_dir=task_request.output_dir
    )
    
    return task

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, summary="查询任务状态", description="查询指定任务的状态")
async def get_task_status(task_id: str):
    """
    查询指定任务的状态
    """
    # 检查任务是否存在
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    # 获取任务
    task = tasks[task_id]
    
    # 构建响应
    response = TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        message=task.message,
        progress=task.progress
    )
    
    # 如果任务已完成，添加结果报告路径
    if task.status == TaskStatus.COMPLETED and task.output_dir:
        combined_report_path = os.path.join(task.output_dir, "combined_report.json")
        if os.path.exists(combined_report_path):
            response.result_path = combined_report_path
    
    return response

@app.delete("/tasks/{task_id}", response_model=TaskStatusResponse, summary="关闭任务", description="关闭指定的任务")
async def cancel_task(task_id: str):
    """
    关闭指定的任务
    """
    # 检查任务是否存在
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    # 获取任务
    task = tasks[task_id]
    
    # 如果任务正在运行，则标记为已取消
    if task.status == TaskStatus.RUNNING or task.status == TaskStatus.PENDING:
        with task_lock:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            task.message = "任务已取消"
    
    # 构建响应
    response = TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        message=task.message,
        progress=task.progress
    )
    
    return response

@app.get("/", summary="API根路径", description="返回API信息")
async def root():
    """
    返回API信息
    """
    return {
        "name": "数据集质量检查API",
        "version": "1.0.0",
        "description": "提供数据集质量检查服务的API",
        "endpoints": [
            {
                "path": "/tasks",
                "method": "POST",
                "description": "启动评估任务"
            },
            {
                "path": "/tasks/{task_id}",
                "method": "GET",
                "description": "查询任务状态"
            },
            {
                "path": "/tasks/{task_id}",
                "method": "DELETE",
                "description": "关闭任务"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)