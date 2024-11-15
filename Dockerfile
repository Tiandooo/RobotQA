# 使用官方 Python 3.9 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录为 /app
WORKDIR /app

# 复制当前目录下的所有文件到容器内的 /app 目录
COPY . /app

# 安装依赖
RUN apt-get update

RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN rm -rf /var/lib/apt/lists/*

# 创建 pip 配置文件夹
RUN mkdir -p ~/.config/pip

# 设置 pip 的镜像源为清华大学 PyPI 镜像
RUN echo "[global]\nindex-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/" > ~/.config/pip/pip.conf

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 开放端口 8080
EXPOSE 8080

# 运行 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]