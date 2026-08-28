# 使用说明

## 快速启动

### Windows
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=.
python main.py
streamlit run app.py
```

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python3 main.py
streamlit run app.py
```

### Docker
```bash
docker compose up --build
```

浏览器访问：http://localhost:8501

> 说明：系统包含两个进程，需同时启动：
> 1. `python main.py`（FastAPI 后端，端口 8000）
> 2. `streamlit run app.py`（前端界面，端口 8501）

## 配置 API Key

复制 `.example.env` 为 `.env`，填入对应服务商的 API Key：

```bash
cp .example.env .env
# 编辑 .env，填入 API Key
```

## 使用知识库 (RAG)

1. 左侧侧边栏开启「启用知识库问答」
2. 上传 txt / pdf / md / docx 文档
3. 点击「构建知识库」
4. 正常提问，AI 将基于文档内容回答并标注来源

## 切换模型

在左侧侧边栏「模型配置」处选择服务商和模型，无需重启。
