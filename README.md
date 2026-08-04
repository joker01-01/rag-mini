# 最小 RAG 系统（rag-mini）

一套最小可用的检索增强生成（RAG）系统：**任意文件 → Markdown → 按标题切分 → bge-m3 向量化 → Chroma 检索 → DeepSeek 生成**。

## 环境

- Python 3.12+
- 依赖：`pip install -r requirements.txt`
- DeepSeek API key：设置环境变量 `DEEPSEEK_API_KEY`

## 使用

```bash
# 1. 把资料放进 docs/（支持 PDF/DOCX/PPTX/XLSX/MD/TXT/HTML/图片等）
# 2. 构建知识库（首次会自动下载 bge-m3 模型，约 2GB）
python build_kb.py --rebuild

# 3. 问答
python ask.py "产品的退款政策是什么？"

# 4. 生成"裸问 vs RAG"真实对比（用于写作复盘）
python compare.py
```

国内网络下载模型（二选一）：

```powershell
# 方式一：HuggingFace 镜像（可用时）
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 方式二：ModelScope（推荐，国内稳定）
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-m3'))"
$env:EMBED_MODEL = "本地模型下载路径"  # 把上一条命令输出的路径填进来
```

## 参数

- 切分：按 Markdown 标题层级切分，块上限 800 字符（`build_kb.py --max-chars` 可调）
- 检索：top-k = 5，相似度阈值 0.35（`ask.py --top-k --threshold` 可调）
- 生成：DeepSeek `deepseek-chat`，temperature 0.3，强制只依据资料回答并标注引用编号

## 已知局限

- 扫描版 PDF 无法直接提取文字，需要先 OCR
- 纯向量检索对精确关键词（型号、人名）不敏感，可后续加 BM25 混合检索
- 无多轮对话记忆
