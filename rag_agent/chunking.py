"""文档加载与切分模块。

把一篇长文档切成适合检索的片段（chunk）：
1. 先按空行（段落）切分；
2. 超过 max_len 的段落，用「滑动窗口」切到 <= max_len；
3. 窗口步长 = max_len - overlap，相邻片段自然产生 overlap 个字符重叠，
   避免关键信息恰好被切在片段边界处。

面试高频点：为什么要 overlap？—— 防止一个完整语义（比如一句话、一个
公式）被硬切到两个 chunk 里，导致检索时丢失上下文。
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """一个文档片段"""

    text: str  # 片段文本
    source: str  # 来源文件路径
    index: int  # 在来源文件中的片段序号


def load_text(path: str) -> str:
    """读取 txt/md 文本文件，utf-8 返回全文。"""
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_text(text: str, max_len: int = 500, overlap: int = 80) -> list[str]:
    """把长文本切成若干片段。

    Args:
        text: 原始全文
        max_len: 单个片段最大字符数
        overlap: 相邻片段重叠字符数（要求 0 <= overlap < max_len）
    """
    # 1) 先按空行（段落）切分
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 滑动窗口步长
    step = max(1, max_len - overlap)

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_len:
            # 短段落直接作为一个 chunk
            chunks.append(para)
        else:
            # 长段落用滑动窗口切分
            start = 0
            while start < len(para):
                chunks.append(para[start : start + max_len])
                if start + max_len >= len(para):
                    break
                start += step
    return chunks


def load_documents(paths: list[str]) -> list[Chunk]:
    """读取多个文件 → 切分 → 返回 Chunk 列表（source=路径，index 递增）。"""
    all_chunks: list[Chunk] = []
    for path in paths:
        text = load_text(path)
        for i, piece in enumerate(split_text(text)):
            all_chunks.append(Chunk(text=piece, source=path, index=i))
    return all_chunks
