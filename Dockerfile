# 基础镜像：Python 3.11 轻量版（slim 比完整版小很多）
FROM python:3.11-slim

# 设置工作目录（容器内的项目路径）
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖（pdf 解析需要）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先只复制依赖文件（利用 Docker 缓存层，依赖不变时不重新安装）
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p chat_history knowledge_base vector_db .streamlit

# 暴露端口（FastAPI + Streamlit）
EXPOSE 8000 8501

ENV PYTHONPATH=/app

# 健康检查移到 docker-compose（api/web 两个服务各自检查不同端口）
# 默认启动 API；Web 服务通过 docker-compose 的 command 覆盖
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
