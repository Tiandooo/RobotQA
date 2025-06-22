#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据格式检查器的FastAPI接口
"""

import os
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from utils.data_format_checker import DataFormatChecker

app = FastAPI(
    title="数据格式检查API",
    description="用于检查数据集中的图像、视频和运动学/动力学数据的格式是否符合标准要求",
    version="1.0.0"
)

# 存储后台任务的结果
task_results = {}


class NamingRules(BaseModel):
    """命名规则模型"""
    image: Optional[Dict[str, str]] = None
    depth: Optional[Dict[str, str]] = None
    video: Optional[Dict[str, str]] = None
    kinematic: Optional[Dict[str, str]] = None
    dynamic: Optional[Dict[str, str]] = None


class CheckRequest(BaseModel):
    """检查请求模型"""
    dataset_path: str
    naming_rules: Optional[NamingRules] = None


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    message: str


@app.get("/")
async def root():
    """API根路径"""
    return {"message": "数据格式检查API已启动，访问 /docs 查看API文档"}


@app.post("/check", response_model=TaskResponse)
async def check_dataset(request: CheckRequest, background_tasks: BackgroundTasks):
    """
    检查数据集格式
    
    参数:
        request: 包含数据集路径和可选命名规则的请求
        background_tasks: FastAPI后台任务
    
    返回:
        任务ID和状态
    """
    # 检查数据集路径是否存在
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail=f"数据集路径不存在: {request.dataset_path}")
    
    # 生成任务ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # 添加后台任务
    background_tasks.add_task(
        run_check_task,
        task_id=task_id,
        dataset_path=request.dataset_path,
        naming_rules=request.naming_rules.dict() if request.naming_rules else None
    )
    
    return TaskResponse(
        task_id=task_id,
        status="running",
        message="数据格式检查任务已启动"
    )


@app.get("/task/{task_id}", response_model=Dict[str, Any])
async def get_task_result(task_id: str):
    """
    获取任务结果
    
    参数:
        task_id: 任务ID
    
    返回:
        任务结果
    """
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail=f"任务ID不存在: {task_id}")
    
    return task_results[task_id]


@app.post("/upload_and_check")
async def upload_and_check(
    background_tasks: BackgroundTasks,
    dataset_file: UploadFile = File(...),
    naming_rules_json: Optional[str] = Form(None)
):
    """
    上传数据集并检查格式
    
    参数:
        background_tasks: FastAPI后台任务
        dataset_file: 上传的数据集文件（ZIP格式）
        naming_rules_json: 命名规则JSON字符串（可选）
    
    返回:
        任务ID和状态
    """
    # 检查文件类型
    if not dataset_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="只支持ZIP格式的数据集文件")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, dataset_file.filename)
    
    try:
        # 保存上传的文件
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(dataset_file.file, f)
        
        # 解压文件
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extract_dir = os.path.join(temp_dir, "dataset")
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)
        
        # 解析命名规则
        naming_rules = None
        if naming_rules_json:
            try:
                naming_rules = json.loads(naming_rules_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="命名规则JSON格式错误")
        
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # 添加后台任务
        background_tasks.add_task(
            run_check_task,
            task_id=task_id,
            dataset_path=extract_dir,
            naming_rules=naming_rules,
            is_temp_dir=True
        )
        
        return TaskResponse(
            task_id=task_id,
            status="running",
            message="数据集已上传，格式检查任务已启动"
        )
    
    except Exception as e:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"处理上传文件时出错: {str(e)}")


def run_check_task(task_id: str, dataset_path: str, naming_rules: Optional[Dict] = None, is_temp_dir: bool = False):
    """
    运行检查任务
    
    参数:
        task_id: 任务ID
        dataset_path: 数据集路径
        naming_rules: 命名规则
        is_temp_dir: 是否为临时目录
    """
    try:
        # 创建检查器
        checker = DataFormatChecker(dataset_path)
        
        # 设置命名规则
        if naming_rules:
            checker.set_naming_rules(naming_rules)
        
        # 运行检查
        results = checker.check_all()
        
        # 生成报告
        report_path = os.path.join(tempfile.gettempdir(), f"report_{task_id}.json")
        checker.generate_report(report_path)
        
        # 读取报告
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # 存储结果
        task_results[task_id] = {
            "status": "completed",
            "dataset_path": dataset_path,
            "report": report
        }
        
        # 清理临时文件
        os.remove(report_path)
        
        # 如果是临时目录，清理它
        if is_temp_dir:
            shutil.rmtree(dataset_path, ignore_errors=True)
    
    except Exception as e:
        # 存储错误信息
        task_results[task_id] = {
            "status": "failed",
            "dataset_path": dataset_path,
            "error": str(e)
        }
        
        # 如果是临时目录，清理它
        if is_temp_dir:
            shutil.rmtree(dataset_path, ignore_errors=True)


@app.get("/naming_rules/default")
async def get_default_naming_rules():
    """
    获取默认命名规则
    
    返回:
        默认命名规则
    """
    # 创建一个临时检查器以获取默认规则
    temp_checker = DataFormatChecker(".")
    return temp_checker.naming_rules


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='启动数据格式检查API服务')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='主机地址')
    parser.add_argument('--port', type=int, default=8000, help='端口号')
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)