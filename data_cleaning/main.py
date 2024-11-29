from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_cleaning.task_manager import TaskManager

app = FastAPI()
task_manager = TaskManager()

class StartTaskRequest(BaseModel):
    input_path: str
    output_path: str
    cleaning_params: dict

@app.post("/data-cleaning/start")
def start_data_cleaning_task(request: StartTaskRequest):
    task_id = task_manager.start_task("data_cleaning", request.dict())
    print("创建完成！")
    with open("test_main.txt", "w") as f:
        f.write("start")
    return {"task_id": task_id}

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
