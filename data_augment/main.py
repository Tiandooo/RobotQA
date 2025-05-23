from fastapi import FastAPI, HTTPException
from utils.task_manager import TaskManager
from data_augment.data_augment_task import data_augmentation_task

app = FastAPI()
task_manager = TaskManager()

from pydantic import BaseModel

# 定义数据增强参数模型
class AugmentationParams(BaseModel):
    brightness: float  # 亮度参数
    contrast: float    # 对比度参数
    saturation: float  # 饱和度参数
    hue: float         # 色调参数

# 定义启动任务请求体的 Pydantic 模型
class StartAugmentationRequest(BaseModel):
    input_path: str
    output_path: str
    augmentation_count: int
    augmentation_params: AugmentationParams



@app.post("/data-augmentation/start")
def start_augmentation_task(request: StartAugmentationRequest):
    """启动数据增强任务"""
    task_id = task_manager.start_task(
        task_type="data_augment",
        params= request.dict())
    return {"task_id": task_id}

@app.get("/data-augmentation/progress")
def get_augmentation_progress(task_id: str):
    """查询数据增强任务进度"""
    task = task_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {
        "progress": task["progress"],
        "status": task["status"],
        "vis": f"{task['params']['output_path']}/vis",
    }

@app.post("/data-augmentation/stop")
def stop_augmentation_task(task_id: str):
    """中止数据增强任务"""
    success = task_manager.stop_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task not found or already completed.")
    return {"status": "success"}