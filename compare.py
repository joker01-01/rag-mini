# -*- coding: utf-8 -*-
"""生成文章用的真实问答对比：同一问题，裸问 DeepSeek vs 走 RAG。

结果写入 results/compare.md，可直接摘录进公众号文章。
"""

import os
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ask import MODEL_NAME, ask, load_kb

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

QUESTIONS = [
    "轻雀知识库支持导入哪些格式的文件？",
    "产品的退款政策是什么？",
    "单个文件的大小限制是多少？",
    "支持离线使用吗？怎么开启？",
    "如何取消订阅？",
    "知识库的权限角色有哪几种？",
    "导入失败一般是什么原因？",
    "团队版包含几个成员？",
]


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

    print("加载模型与向量库 ...")
    model = SentenceTransformer(MODEL_NAME)
    collection = load_kb()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# 实测对比：裸问 DeepSeek vs 走 RAG", ""]
    for q in QUESTIONS:
        print(f"处理问题: {q}")
        bare, _ = ask(q, client, model, collection, use_rag=False)
        rag, hits = ask(q, client, model, collection, use_rag=True)
        lines += [
            f"## 问题：{q}",
            "",
            "### 裸问 DeepSeek",
            "",
            f"> {bare}",
            "",
            "### 走 RAG 的回答",
            "",
            f"> {rag}",
            "",
            "### 检索到的资料片段",
            "",
        ]
        for i, h in enumerate(hits, 1):
            src = h["meta"].get("source", "")
            head = h["meta"].get("heading", "")
            lines.append(f"- [{i}] {src} / {head}（相似度 {h['score']}）")
        lines.append("")
        lines.append("---")
        lines.append("")

    out = RESULTS_DIR / "compare.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"对比结果已写入: {out}")


if __name__ == "__main__":
    main()
