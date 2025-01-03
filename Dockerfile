# 使用官方 Python 3.9 镜像作为基础镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 设置工作目录为 /app
WORKDIR /app

# 复制当前目录下的所有文件到容器内的 /app 目录
COPY . /app


# 覆盖apt源列表为清华大学镜像源
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && apt-get update


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0

# 创建 pip 配置文件夹
RUN mkdir -p ~/.config/pip

# 设置 pip 的镜像源为清华大学 PyPI 镜像
RUN echo "[global]\nindex-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/" > ~/.config/pip/pip.conf

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 开放端口 34809
EXPOSE 34809

# 运行 FastAPI 应用

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "34809", "--reload"]

