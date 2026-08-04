# -*- coding: utf-8 -*-
"""本地检索质量自测（不需要网络，也不需要 API key）。"""

import os

from sentence_transformers import SentenceTransformer

from ask import load_kb, retrieve

QUESTIONS = [
    "产品的退款政策是什么？",
    "单个文件的大小限制是多少？",
    "知识库的权限角色有哪几种？",
    "支持离线使用吗？",
    "如何取消订阅？",
    "导入失败一般是什么原因？",
    "团队版包含几个成员？",
    "问答时最多引用几个资料片段？",
]


def main() -> None:
    model = SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-m3"))
    collection = load_kb()
    for q in QUESTIONS:
        hits = retrieve(q, model, collection)
        print(f"Q: {q}")
        for i, h in enumerate(hits, 1):
            src = h["meta"].get("source", "")
            head = h["meta"].get("heading", "")
            print(f"  [{i}] sim={h['score']} | {src} | {head}")
        print()


if __name__ == "__main__":
    main()
