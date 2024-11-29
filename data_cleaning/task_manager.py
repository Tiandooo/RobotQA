import threading
from typing import Dict, Any
import uuid
import time


class TaskManager:
    def __init__(self):
        self.tasks = {}  # 存储任务信息，key为task_id

    def start_task(self, task_type: str, params: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        task = {
            "type": task_type,
            "status": "running",
            "progress": 0,
            "params": params,
            "result": None,
            "thread": None,
            "stop_event": None,
        }
        stop_event = threading.Event()  # 每个任务对应一个终止事件
        # 启动任务线程
        thread = threading.Thread(target=self._run_task, args=(task_id, stop_event))
        task["thread"] = thread
        task["stop_event"] = stop_event
        self.tasks[task_id] = task
        thread.start()
        return task_id

    def _run_task(self, task_id: str, stop_event: threading.Event):
        task = self.tasks[task_id]
        task_type = task["type"]
        params = task["params"]

        # 根据任务类型调用相应的执行逻辑
        if task_type == "data_cleaning":
            from data_cleaning.data_cleaning_task import run_data_cleaning
            task["result"] = run_data_cleaning(params, lambda p: self.update_progress(task_id, p), stop_event)
            if task["result"]["status"] != "paused":
                task["status"] = "completed"
                task["progress"] = 100

    def update_progress(self, task_id: str, progress: int):
        if task_id in self.tasks and self.tasks[task_id]["status"] == "running":
            self.tasks[task_id]["progress"] = progress

    def stop_task(self, task_id: str):
        task = self.tasks.get(task_id)
        if task and task["status"] == "running":
            task["stop_event"].set()  # 设置终止信号
            task["thread"].join()  # 等待线程结束
            task["status"] = "stopped"
            return True
        return False

    def get_task_status(self, task_id: str):
        return self.tasks.get(task_id, None)
