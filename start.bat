@echo off
chcp 65001 >nul
:: Windows 服务器一键启动脚本
:: 双击运行或在命令行执行

echo ========================================
echo   AI 问答系统 启动中...
echo ========================================

:: 进入脚本所在目录
cd /d "%~dp0"

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\activate" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境
echo [2/3] 激活虚拟环境...
call venv\Scripts\activate

:: 安装依赖
echo [3/3] 检查依赖...
pip install -r requirements.txt -q

:: 检查 .env 文件
if not exist ".env" (
    echo [警告] 未找到 .env 文件，正在从模板创建...
    copy .env.example .env
    echo [警告] 请编辑 .env 文件填入你的 API Key
    notepad .env
)

echo.
echo 启动 Streamlit 服务...
echo 访问地址: http://localhost:8501
echo.

:: 启动 Streamlit
set PYTHONPATH=.
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0

pause
