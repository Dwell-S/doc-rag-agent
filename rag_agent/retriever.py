"""BM25 检索模块。

BM25 是一种经典的基于关键词的检索算法，这里用 jieba 做中文分词，
再用 rank_bm25 提供的 BM25Okapi 实现。相比向量检索，它的优点是不需要
embedding 模型、不依赖外部服务，适合作为 RAG 的第一版基线。
"""

import jieba
from rank_bm25 import BM25Okapi

from .chunking import Chunk

# 关闭 jieba 的启动日志，保持控制台干净
jieba.setLogLevel(20)


class BM25Retriever:
    """基于 BM25 的文档片段检索器。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # 对每个 chunk 分词，构建 BM25 索引
        tokenized = [list(jieba.cut(c.text)) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if chunks else None

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        """返回与 query 最相关的 top_k 个 Chunk（按相关度降序）。"""
        if self._bm25 is None or not self.chunks:
            return []

        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)

        # 取分数最高的 top_k 个索引
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.chunks[i] for i in ranked]
