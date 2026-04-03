import subprocess
import sys
import os
import time

def run_local():
    """
    本地一键启动：同时启动 FastAPI 后端和 Streamlit 前端。
    使用 subprocess 并发运行两个进程。
    """
    # 1. 设置 PYTHONPATH（确保能找到 src 目录）
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    # 2. 启动 FastAPI (后端)
    print("🚀 正在启动 FastAPI 后端服务 (端口 8000)...")
    # 使用 sys.executable 确保使用当前虚拟环境的 python 解释器
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        env=env
    )

    # 等待几秒，让后端先初始化模型
    time.sleep(3)

    # 3. 启动 Streamlit (前端)
    print("🎨 正在启动 Streamlit 前端界面 (端口 8501)...")
    try:
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port", "8501"],
            env=env
        )

        print("\n✅ 服务已就绪！")
        print("🔗 前端访问: http://localhost:8501")
        print("🔗 API 文档: http://localhost:8000/docs")
        print("\n按 Ctrl+C 停止所有服务...\n")

        # 保持主进程运行，直到用户按下 Ctrl+C
        api_process.wait()
        streamlit_process.wait()

    except KeyboardInterrupt:
        print("\n🛑 正在关闭所有服务...")
        api_process.terminate()
        streamlit_process.terminate()
        print("✅ 已安全关闭")

if __name__ == "__main__":
    # 检查依赖
    try:
        import fastapi
        import streamlit
        import uvicorn
    except ImportError:
        print("❌ 缺少必要依赖！请运行: pip install -r requirements.txt")
        sys.exit(1)

    run_local()
