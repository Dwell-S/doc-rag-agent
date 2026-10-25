"""tools 模块测试：验证知识库与工具封装。"""

from rag_agent.tools import KnowledgeBase, build_tools


def test_build_tools_returns_three_tools():
    kb = KnowledgeBase()
    tools = build_tools(kb)
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"search_documents", "load_documents", "list_documents"}


def test_load_and_search_tools():
    kb = KnowledgeBase()
    tools = build_tools(kb)
    by_name = {t.name: t for t in tools}

    # 加载文档
    load_result = by_name["load_documents"].execute({"paths": "data/sample.md"})
    assert load_result.success is True
    assert "片段" in load_result.result

    # 检索
    search_result = by_name["search_documents"].execute(
        {"query": "RAG 是什么", "top_k": 2}
    )
    assert search_result.success is True
    assert "RAG" in search_result.result or "检索" in search_result.result

    # 列表
    list_result = by_name["list_documents"].execute({})
    assert list_result.success is True
    assert "data/sample.md" in list_result.result


def test_search_empty_kb_returns_hint():
    kb = KnowledgeBase()
    tools = build_tools(kb)
    search = {t.name: t for t in tools}["search_documents"]
    result = search.execute({"query": "随便问"})
    assert result.success is True
    assert "加载文档" in result.result
