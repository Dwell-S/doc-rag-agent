"""检索管线演示 —— 无需 API Key，纯本地运行。

展示 RAG 的「切分 → 检索」两步：在没有大模型 key 的情况下，
也能直观看到「文档被切成什么样子、检索命中了哪些片段」。

用法（在项目根目录 doc-rag-agent 下）:
    python -m rag_agent.demo
"""

import os
import sys

from .chunking import load_documents
from .retriever import BM25Retriever


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data_dir = os.path.join(os.getcwd(), "data")
    paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith((".md", ".txt"))
    ]
    if not paths:
        print("data/ 目录下没有文档，请先放一些 .md/.txt 文件。")
        return

    chunks = load_documents(paths)
    print(f"已加载 {len(paths)} 个文件，切分成 {len(chunks)} 个片段：\n")
    for c in chunks:
        print(f"[{c.source} #{c.index}] ({len(c.text)} 字) {c.text[:45]}...")

    retriever = BM25Retriever(chunks)
    queries = [
        "DeepSeek 的 API 价格是多少",
        "什么是 RAG",
        "为什么 DeepSeek 适合做 Agent",
    ]
    for q in queries:
        print("\n" + "=" * 50)
        print(f"🔍 问题：{q}")
        print("=" * 50)
        for i, c in enumerate(retriever.search(q, top_k=2), 1):
            print(f"\n  Top{i}  [{c.source} #{c.index}]")
            print(f"  {c.text[:90]}{'...' if len(c.text) > 90 else ''}")


if __name__ == "__main__":
    main()
