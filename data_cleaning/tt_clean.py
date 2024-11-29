"""
Author: Ruibin Du

测试数据清洗接口

"""
import requests
import time

# 定义接口地址
BASE_URL = "http://127.0.0.1:8000"  # 替换为 FastAPI 服务的实际地址

# 定义任务参数
task_params = {
    "input_path": "data/kitchen_40demo",
    "output_path": "data/kitchen_40demo/clean",
    "cleaning_params": {
        "missing_value": "fill",
        "fill_method": "linear",
        "remove_duplicates": True
    }
}

# 启动任务
start_response = requests.post(f"{BASE_URL}/data-cleaning/start", json=task_params)
if start_response.status_code == 200:
    start_data = start_response.json()
    task_id = start_data["task_id"]
    print(f"Task started successfully! Task ID: {task_id}")
else:
    print(f"Failed to start task: {start_response.status_code}, {start_response.text}")
    exit()

time.sleep(0.1)
# 查询任务进度（无需停顿）
progress_response = requests.get(f"{BASE_URL}/data-cleaning/progress", params={"task_id": task_id})
if progress_response.status_code == 200:
    progress_data = progress_response.json()
    print(f"1 Task Progress: {progress_data}")
else:
    print(f"Failed to query progress: {progress_response.status_code}, {progress_response.text}")

# time.sleep(0.1)
# 查询任务进度（无需停顿）
progress_response = requests.get(f"{BASE_URL}/data-cleaning/progress", params={"task_id": task_id})
if progress_response.status_code == 200:
    progress_data = progress_response.json()
    print(f"2 Task Progress: {progress_data}")
else:
    print(f"Failed to query progress: {progress_response.status_code}, {progress_response.text}")

time.sleep(0.1)
# 终止任务
stop_response = requests.post(f"{BASE_URL}/data-cleaning/stop", params={"task_id": task_id})
if stop_response.status_code == 200:
    stop_data = stop_response.json()
    print(f"Task stopped successfully! Status: {stop_data['status']}")
else:
    print(f"Failed to stop task: {stop_response.status_code}, {stop_response.text}")

# 再次查询任务进度
progress_response_after_stop = requests.get(f"{BASE_URL}/data-cleaning/progress", params={"task_id": task_id})
if progress_response_after_stop.status_code == 200:
    progress_data_after_stop = progress_response_after_stop.json()
    print(f"Task Progress after stop: {progress_data_after_stop}")
else:
    print(f"Failed to query progress after stop: {progress_response_after_stop.status_code}, {progress_response_after_stop.text}")
