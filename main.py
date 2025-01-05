from typing import Optional, Union, Tuple

from fastapi import FastAPI, HTTPException
from utils.task_manager import TaskManager
from data_augment.data_augment_task import data_augmentation_task

app = FastAPI()
task_manager = TaskManager()

from pydantic import BaseModel, Field


# 定义数据增强参数模型
class AugmentationParams(BaseModel):
    """
    数据增强参数配置类。
    支持多种图像增强操作，包括亮度、对比度、饱和度、色调等。
    每个参数可以是单个值（在值附近波动）或范围（在范围内随机生成）。
    """
    brightness: Optional[Union[float, Tuple[float, float]]] = Field(
        default=None,
        description="亮度调整参数。可以是单个值（如 1.2）或范围（如 [0.8, 1.5]）。"
                    "如果为单个值，则在该值附近小幅度波动（默认波动范围为 ±10%）。",
        example=1.2
    )
    contrast: Optional[Union[float, Tuple[float, float]]] = Field(
        default=None,
        description="对比度调整参数。可以是单个值（如 1.2）或范围（如 [0.8, 1.5]）。"
                    "如果为单个值，则在该值附近小幅度波动（默认波动范围为 ±10%）。",
        example=[0.8, 1.5]
    )
    saturation: Optional[Union[float, Tuple[float, float]]] = Field(
        default=None,
        description="饱和度调整参数。可以是单个值（如 1.2）或范围（如 [0.8, 1.5]）。"
                    "如果为单个值，则在该值附近小幅度波动（默认波动范围为 ±10%）。",
        example=None
    )
    hue: Optional[Union[float, Tuple[float, float]]] = Field(
        default=None,
        description="色调调整参数。可以是单个值（如 4）或范围（如 [-5, 5]）。"
                    "如果为单个值，则在该值附近小幅度波动（默认波动范围为 ±10%）。",
        example=4
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "brightness": 1.2,
                "contrast": [0.8, 1.5],
                "saturation": None,
                "hue": 4
            }
        }
    }


# 定义启动任务请求体的 Pydantic 模型
class StartAugmentationRequest(BaseModel):
    """
    启动数据增强任务的请求体模型。
    """
    input_path: str = Field(
        ...,
        description="输入图像的根目录路径。",
        example="/path/to/input"
    )
    output_path: str = Field(
        ...,
        description="输出图像的根目录路径。",
        example="/path/to/output"
    )
    augmentation_count: int = Field(
        ...,
        description="每张图像的增强数量。",
        example=5
    )
    augmentation_params: AugmentationParams = Field(
        ...,
        description="数据增强参数配置。"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "input_path": "/path/to/input",
                "output_path": "/path/to/output",
                "augmentation_count": 5,
                "augmentation_params": {
                    "brightness": 1.2,
                    "contrast": [0.8, 1.5],
                    "saturation": None,
                    "hue": 4
                }
            }
        }
    }


@app.post("/data-augmentation/start")
def start_augmentation_task(request: StartAugmentationRequest):
    """启动数据增强任务"""
    # task_id = task_manager.start_task(
    #     task_type="data_augment",
    #     params= request.dict())
    # return {"task_id": task_id}
    return task_manager.start_task(
        task_type="data_augment",
        params=request.dict())


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


"""
数据清洗
"""


class StartTaskRequest(BaseModel):
    input_path: str
    output_path: str
    cleaning_params: dict


@app.post("/data-cleaning/start")
def start_data_cleaning_task(request: StartTaskRequest):
    return task_manager.start_task("data_cleaning", request.dict())
    # print("创建完成！")
    #
    # return {"task_id": task_id}


@app.post("/data-cleaning/stop")
def stop_data_cleaning_task(task_id: str):
    success = task_manager.stop_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task not found or already completed.")
    return {"status": "success"}


@app.get("/data-cleaning/progress")
def get_data_cleaning_progress(task_id: str):
    task = task_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {
        "progress": task["progress"],
        "status": task["status"],
        "vis": f"{task['params']['output_path']}/vis",
    }


"""
数据质量评估
"""

# 定义数据质量评估参数模型
class QAParams(BaseModel):
    completeness_score: bool = False  # 完整性得分
    accuracy_score: bool = True  # 准确性得分
    consistency_score: bool = False # 一致性得分
    accessable_score: bool = False # 可达性得分


# 定义启动任务请求体的 Pydantic 模型
class StartQARequest(BaseModel):
    input_path: str
    output_path: str
    qa_params: QAParams
    device: str


@app.post("/data-qa/start")
def start_qa_task(request: StartQARequest):
    """启动数据质量评估任务"""
    return task_manager.start_task(
        task_type="data_qa",
        params=request.dict())
    # return {"task_id": task_id}


@app.get("/data-qa/progress")
def get_qa_progress(task_id: str):
    """查询数据质量评估任务进度"""
    task = task_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {
        "progress": task["progress"],
        "status": task["status"],
        "vis": f"{task['params']['output_path']}/vis",
    }


@app.post("/data-qa/stop")
def stop_qa_task(task_id: str):
    """中止数据质量评估任务"""
    success = task_manager.stop_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task not found or already completed.")
    return {"status": "success"}
