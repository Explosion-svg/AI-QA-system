# 使用说明

## 快速启动

### Windows
```bat
双击 start.bat
```
或手动：
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=.
streamlit run src/app.py
```

### Linux / Mac
```bash
chmod +x start.sh && ./start.sh
```
或手动：
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
streamlit run src/app.py
```

### Docker
```bash
docker-compose up --build
```

浏览器访问：http://localhost:8501

## 配置 API Key

复制 `.example.env` 为 `.env`，填入对应服务商的 API Key：

```bash
cp .example.env .env
# 编辑 .env，填入 API Key
```

## 使用知识库 (RAG)

1. 左侧侧边栏开启「启用知识库问答」
2. 上传 txt / pdf / md 文档
3. 点击「构建知识库」
4. 正常提问，AI 将基于文档内容回答并标注来源

## 切换模型

在左侧侧边栏「模型配置」处选择服务商和模型，无需重启。
