# -*- coding: utf-8 -*-
"""最小 RAG 系统 - 问答

问题 -> bge-m3 向量化 -> Chroma 检索 top-k -> 拼上下文 -> DeepSeek 生成（带引用编号）
"""

import argparse
import os
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "data" / "kb"
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
TOP_K = 5
SIM_THRESHOLD = 0.35
SYSTEM_PROMPT = (
    "你是一个严格基于资料回答问题的助手。"
    "只依据【资料】中的内容回答，资料里没有的信息，明确回答'资料中没有相关内容'，不要编造。"
    "回答末尾用 [1][2] 形式标注引用的资料编号。"
)


def load_kb():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("rag_kb", metadata={"hnsw:space": "cosine"})
    return collection


def retrieve(query: str, model, collection, top_k: int = TOP_K, threshold: float = SIM_THRESHOLD):
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1.0 - dist  # cosine distance -> similarity
        if sim >= threshold:
            hits.append({"doc": doc, "score": round(sim, 3), "meta": meta})
    return hits


def ask(question: str, client: OpenAI, model, collection, use_rag: bool = True):
    if use_rag:
        hits = retrieve(question, model, collection)
        if not hits:
            return "（未检索到相似度达标的资料片段）", []
        parts = [
            f"[{i}] 来源: {h['meta'].get('source', '')} / {h['meta'].get('heading', '')}\n{h['doc']}"
            for i, h in enumerate(hits, 1)
        ]
        context = "\n\n".join(parts)
        user_content = f"【资料】\n{context}\n\n【问题】\n{question}"
    else:
        hits = []
        user_content = question

    resp = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content, hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="要问的问题；不填则进入交互模式")
    parser.add_argument("--no-rag", action="store_true", help="裸问 DeepSeek，不检索资料")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

    print(f"加载 embedding 模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    collection = load_kb()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def run(q: str):
        answer, hits = ask(q, client, model, collection, use_rag=not args.no_rag)
        print(f"\n问题: {q}")
        print(f"回答: {answer}")
        if hits:
            print("检索片段:")
            for i, h in enumerate(hits, 1):
                head = h["meta"].get("heading", "") or h["doc"][:30]
                print(f"  [{i}] sim={h['score']} | {h['meta'].get('source')} | {head}")

    if args.question:
        run(args.question)
    else:
        while True:
            q = input("问题（输入 exit 退出）: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q:
                run(q)


if __name__ == "__main__":
    main()
