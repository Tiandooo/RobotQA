### 1. 数据清洗模块

#### 1.1 启动数据清洗任务

- **URL**: `/data-cleaning/start`
- **Method**: `POST`
- **Input**:
  - `input_path`: 输入数据路径
  - `output_path`: 输出数据路径
    - vis: 包含删掉的图片
    - result：清洗结果
  - `cleaning_params`: 清洗参数（如去重、缺失值处理等）
    - 缺失值，删除
- **Output**:
  - `task_id`: 任务ID





#### 1.2 中止数据清洗任务

- **URL**: `/data-cleaning/stop`
- **Method**: `POST`
- **Input**:
  - `task_id`: 任务ID
- **Output**:
  - `status`: 中止状态（成功/失败）





#### 1.3 查询数据清洗任务进度（图像）

- **URL**: `/data-cleaning/progress`
- **Method**: `GET`
- **Input**:
  - `task_id`: 任务ID
- **Output**:
  - `progress`: 任务进度（百分比）
  - `status`: 任务状态（运行中/已完成/已中止）
  - `vis` 错误的结果存放另一个文件夹，返回路径
  - `msg`: 删除的数据返回





### 2. 数据质量评估模块

#### 2.1 启动数据质量评估任务

- **URL**: `/data-quality/start`
- **Method**: `POST`
- **Input**:
  - `input_path`: 输入数据路径
  - `output_path`: 输出数据路径
  - `evaluation_params`: 评估参数（如完整性、一致性等）
    - image_quality 
    - **and so on**
- **Output**:
  - `task_id`: 任务ID

#### 2.2 中止数据质量评估任务

- **URL**: `/data-quality/stop`
- **Method**: `POST`
- **Input**:
  - `task_id`: 任务ID
- **Output**:
  - `status`: 中止状态（成功/失败）

#### 2.3 查询数据质量评估任务进度

- **URL**: `/data-quality/progress`
- **Method**: `GET`
- **Input**:
  - `task_id`: 任务ID
- **Output**:
  - `progress`: 任务进度（百分比）
  - `status`: 任务状态（运行中/已完成/已中止）
  - `result`：质量评估得分（已完成提供）

### 3. 数据增强模块

#### 3.1 启动数据增强任务

- **URL**: `/data-augmentation/start`
- **Method**: `POST`
- **Input**:
  - `input_path`: 输入数据路径
  - `output_path`: 输出数据路径
  - `augmentation_params`: 增强参数
    - `brightness`: 亮度
    - `contrast`: 对比度
    - `saturation`: 饱和度
    - `hue`：色调
- **Output**:
  - `task_id`: 任务ID

#### 3.2 中止数据增强任务

- **URL**: `/data-augmentation/stop`
- **Method**: `POST`
- **Input**:
  - `task_id`: 任务ID
- **Output**:
  - `status`: 中止状态（成功/失败）

#### 3.3 查询数据增强任务进度

- **URL**: `/data-augmentation/progress`
- **Method**: `GET`
- **Input**:
  - `task_id`: 任务ID
- **Output**:
  - `progress`: 任务进度（百分比）
  - `status`: 任务状态（运行中/已完成/已中止）

