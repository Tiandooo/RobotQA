
# 环境

下载权重文件，放在data_qa/pretrained_weights文件夹中。
 [model](https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth) 



# 安装所需库
```bash
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install -r requirements_backup.txt
```

# 运行fastapi接口
```bash
uvicorn main:app --host=0.0.0.0 --port 8022 --reload
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

POST http://127.0.0.1:8022/data-qa/start
```json

{
    "input_path": "data/kitchen_40demo/raw_data",
    "output_path": "data/kitchen_40demo/qa",
    "device": "cpu",
    "qa_params": {
        "video": true
        
    }
  
}
```

GET http://127.0.0.1:8022/data-qa/progress

params: task_id


POST http://127.0.0.1:8022/data-qa/stop

params: task_id




nvidia-docker

docker run -p 34809:34809 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/server/RobotQA/data:/root/Documents/data:rw --rm pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime /bin/bash


docker run -p 34809:34809 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it robot_gov /bin/bash

sudo docker run -d -p 34809:34809 --gpus all -it -v /home/server/fdu/data:/app/data my_robot_dataset_gov



临时退出docker container
ctrl+p ctrl+q

进入docker container
sudo docker exec -it efce35a56f90 bash


sudo docker commit {id} {image_name} 

docker 

![img_1.png](img_1.png)

![img_2.png](img_2.png)