
# 环境

下载权重文件，放在data_qa/pretrained_weights文件夹中。
 [model](https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth) 

# 安装所需库
```bash
pip install -r requirements.txt
```

# 运行fastapi接口
```bash
uvicorn main:app --host= 0.0.0.0 --port 8022 --reload
```


# 数据增强接口样例：

POST http://127.0.0.1:8022/data-augmentation/start
```json

{
    "input_path": "data/kitchen_40demo/raw_data",
    "output_path": "data/kitchen_40demo/aug",
    "augmentation_params": {
        "brightness": true,
        "contrast": true,
        "saturation": false,
        "hue": true
    },
    "augmentation_count": 3
}


```

GET http://127.0.0.1:8022/data-augmentation/progress
params: task_id


POST http://127.0.0.1:8022/data-augmentation/stop
params: task_id

# 数据清洗接口样例：
missing_value可以设置为delete或者fill

POST http://127.0.0.1:8022/data-cleaning/start
```json

{
  "input_path": "data/kitchen_40demo/raw_data",
  "output_path": "data/kitchen_40demo/aug",
  "cleaning_params": {
    "missing_value": "delete"
  }
}


```

GET http://127.0.0.1:8022/data-cleaning/progress

params: task_id


POST http://127.0.0.1:8022/data-cleaning/stop

params: task_id


# 数据质量评估接口样例：

git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data_qa/pretrained_weights/FAST_VQA_B_1_4.pth"  --prune-empty --tag-name-filter cat -- --all 
POST http://127.0.0.1:8022/data-qa/start
```json

{
    "input_path": "data/kitchen_40demo/raw_data",
    "output_path": "data/kitchen_40demo/qa",
    "device": "cuda",
    "qa_params": {
        "video": true
        
    }
  
}
```

GET http://127.0.0.1:8022/data-qa/progress

params: task_id


POST http://127.0.0.1:8022/data-qa/stop

params: task_id
