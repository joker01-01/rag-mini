# -*- coding: utf-8 -*-
"""最小 RAG 系统 - 知识库构建

任意文件 -> Markdown (MarkItDown) -> 按标题切分 -> bge-m3 向量化 -> Chroma 入库
"""

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "data" / "kb"
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xls", ".md", ".txt", ".html", ".jpg", ".jpeg", ".png"}


def convert_to_markdown(path: Path) -> str:
    md = MarkItDown()
    result = md.convert(str(path))
    return result.text_content or ""


def chunk_by_heading(md_text: str, max_chars: int = 800) -> list[dict]:
    """按 Markdown 标题层级切分；超长章节按长度硬切，chunk 保留标题作为上下文前缀。"""
    chunks = []
    heading_stack: list[str] = []
    current_head = ""
    buf = ""

    def make_text(prefix: str, body: str) -> str:
        body = body.strip()
        return f"{prefix}\n\n{body}" if prefix and body else body

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append({"heading": current_head, "text": make_text(current_head, buf)})
        buf = ""

    for line in md_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_head = " > ".join(heading_stack)
            buf = line + "\n"
        else:
            buf += line + "\n"
            while len(buf) > max_chars:
                cut = max(buf.rfind("\n", 0, max_chars), 1)
                chunks.append({"heading": current_head, "text": make_text(current_head, buf[:cut])})
                buf = buf[cut:].lstrip("\n")
    flush()
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--rebuild", action="store_true", help="清空已有向量库后重建")
    args = parser.parse_args()

    print(f"加载 embedding 模型: {args.model}")
    model = SentenceTransformer(args.model)

    import chromadb

    if args.rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("rag_kb", metadata={"hnsw:space": "cosine"})

    entries = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            print(f"转换: {path.name}")
            md_text = convert_to_markdown(path)
            chunks = chunk_by_heading(md_text, max_chars=args.max_chars)
            print(f"  -> Markdown {len(md_text)} 字符, {len(chunks)} 个块")
            for i, c in enumerate(chunks):
                cid = hashlib.md5(f"{path.name}:{i}:{c['text'][:60]}".encode("utf-8")).hexdigest()
                entries.append((cid, c, path.name))

    if not entries:
        print("docs/ 下没有找到可转换的文档")
        return

    texts = [c["text"] for _, c, _ in entries]
    print(f"向量化 {len(texts)} 个块 ...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    ids = [e[0] for e in entries]
    documents = [e[1]["text"] for e in entries]
    metadatas = [
        {"source": e[2], "heading": e[1]["heading"]} for e in entries
    ]
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings.tolist(), metadatas=metadatas)
    print(f"入库完成: {len(ids)} 个块 -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
