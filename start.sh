#!/bin/bash
# Linux/Mac 服务器一键启动脚本
# 使用方法：chmod +x start.sh && ./start.sh

set -e  # 任何命令失败立即退出

echo "========================================"
echo "  AI 问答系统 启动中..."
echo "========================================"

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "[1/3] 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "[2/3] 激活虚拟环境..."
source venv/bin/activate

# 安装/更新依赖
echo "[3/3] 检查依赖..."
pip install -r requirements.txt -q

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "[警告] 未找到 .env 文件，正在从模板创建..."
    cp .env.example .env
    echo "[警告] 请编辑 .env 文件填入你的 API Key"
fi

echo ""
echo "启动 Streamlit 服务..."
echo "访问地址: http://$(hostname -I | awk '{print $1}'):8501"
echo ""

# 启动 Streamlit
streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
