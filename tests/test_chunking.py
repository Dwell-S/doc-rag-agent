"""chunking 模块测试。"""

from rag_agent.chunking import Chunk, load_documents, split_text


def test_split_text_short_paragraph_unchanged():
    text = "这是一段很短的文本。"
    assert split_text(text) == [text]


def test_split_text_long_paragraph_respects_max_len():
    long_text = "这是一个测试句子。" * 100  # 1000 字符
    chunks = split_text(long_text, max_len=120, overlap=20)
    assert len(chunks) > 1
    # 每个片段长度不超过 max_len
    for c in chunks:
        assert len(c) <= 120


def test_split_text_has_overlap():
    # 构造可区分字符，验证相邻片段确实有 overlap 个字符重叠
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    chunks = split_text(text, max_len=100, overlap=20)
    assert len(chunks) > 1
    # 第 i+1 个片段的开头 == 第 i 个片段的末尾 20 个字符
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].startswith(chunks[i][-20:])


def test_load_documents_builds_chunks_with_source_and_index():
    chunks = load_documents(["data/sample.md"])
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "data/sample.md" for c in chunks)
    # index 从 0 开始连续递增
    assert [c.index for c in chunks] == list(range(len(chunks)))
