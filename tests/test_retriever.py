"""retriever 模块测试。"""

from rag_agent.chunking import load_documents
from rag_agent.retriever import BM25Retriever


def test_retriever_returns_relevant_chunk():
    chunks = load_documents(["data/sample.md"])
    r = BM25Retriever(chunks)
    results = r.search("DeepSeek 的 API 价格", top_k=1)
    assert len(results) == 1
    # 命中的片段应包含 API 相关内容
    assert ("API" in results[0].text) or ("base_url" in results[0].text)


def test_retriever_empty_kb_returns_empty():
    r = BM25Retriever([])
    assert r.search("任意问题") == []


def test_retriever_respects_top_k():
    chunks = load_documents(["data/sample.md"])
    r = BM25Retriever(chunks)
    results = r.search("DeepSeek", top_k=2)
    assert len(results) <= 2
