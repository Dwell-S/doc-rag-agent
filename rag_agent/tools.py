"""Agent 工具层：用 @function_tool 把「文档加载 / 检索 / 列表」封装成
Agent 可以调用的工具。

这里体现了 agent 应用开发的关键一步：把业务能力（RAG 检索）抽象成
function calling 工具，让 LLM 在需要时自主调用。
"""

from collections import Counter

from zipagent import function_tool

from .chunking import Chunk, load_documents as _load_docs
from .retriever import BM25Retriever


class KnowledgeBase:
    """文档知识库：持有已加载的片段，并提供检索能力。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._retriever: BM25Retriever | None = None

    def load(self, paths: list[str]) -> int:
        """加载文档，返回新增片段数量。"""
        new_chunks = _load_docs(paths)
        self._chunks.extend(new_chunks)
        self._retriever = BM25Retriever(self._chunks)
        return len(new_chunks)

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        """检索最相关的片段。"""
        if self._retriever is None:
            return []
        return self._retriever.search(query, top_k)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks


def build_tools(kb: KnowledgeBase):
    """构建绑定到指定知识库的 Agent 工具列表。

    说明：@function_tool 会把函数转换成 zipagent.Tool，它会读取函数的
    类型注解和 docstring 自动生成 JSON Schema。因此 docstring 里必须写清
    楚每个参数的含义，LLM 才知道怎么传参。
    """

    @function_tool
    def search_documents(query: str, top_k: int = 3) -> str:
        """在已加载的文档中检索与问题最相关的片段。

        Args:
            query: 要检索的问题或关键词
            top_k: 返回的片段数量，默认 3
        """
        chunks = kb.search(query, top_k)
        if not chunks:
            return "知识库为空或没有检索到相关内容，请先用 load_documents 加载文档。"
        parts = []
        for c in chunks:
            parts.append(f"[来源: {c.source} 片段#{c.index}]\n{c.text}")
        return "\n\n---\n\n".join(parts)

    @function_tool
    def load_documents(paths: str) -> str:
        """加载本地文档到知识库，多个路径用英文逗号分隔。

        Args:
            paths: 文档路径，多个用英文逗号分隔，例如 "data/a.md,data/b.txt"
        """
        path_list = [p.strip() for p in paths.split(",") if p.strip()]
        if not path_list:
            return "未提供有效的文件路径。"
        count = kb.load(path_list)
        return f"成功加载 {len(path_list)} 个文件，新增 {count} 个片段。"

    @function_tool
    def list_documents() -> str:
        """列出知识库中已加载的文档及各文档片段数量。"""
        if not kb.chunks:
            return "知识库为空。"
        counts = Counter(c.source for c in kb.chunks)
        lines = [f"- {src}: {n} 个片段" for src, n in counts.items()]
        return "知识库内容:\n" + "\n".join(lines)

    return [search_documents, load_documents, list_documents]
