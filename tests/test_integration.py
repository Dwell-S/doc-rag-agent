"""端到端集成测试：用脚本化假模型验证「模型调工具 → 真实检索 → 回答」完整链路。

这里不调用真实 LLM，而是用一个按脚本返回响应的假模型，模拟模型的两次行为：
1. 第一次：模型决定调用 search_documents 工具；
2. 第二次：模型拿到检索结果后给出最终回答。

从而验证 Runner 的工具分发、真实 BM25 检索、上下文拼接整条链路是通的。
"""

from zipagent import Agent, Runner
from zipagent.model import Model, ModelResponse, Usage

from rag_agent.tools import KnowledgeBase, build_tools


class ScriptedModel(Model):
    """按脚本顺序返回预设响应的假模型。"""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)

    def generate(self, messages, tools=None):
        return self._responses.pop(0)


def _response(content, tool_calls=None, finish_reason="stop"):
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        usage=Usage(),
        finish_reason=finish_reason,
    )


def test_full_rag_loop_calls_tool_and_answers():
    kb = KnowledgeBase()
    kb.load(["data/sample.md"])

    resp1 = _response(
        content="",
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_documents",
                "arguments": '{"query": "RAG 是什么", "top_k": 2}',
            },
        }],
        finish_reason="tool_calls",
    )
    resp2 = _response(content="RAG 是检索增强生成，可以显著减少幻觉。")
    model = ScriptedModel([resp1, resp2])

    agent = Agent(
        name="TestRAG",
        instructions="先用 search_documents 检索再回答",
        model=model,
        tools=build_tools(kb),
    )

    result = Runner.run(
        agent,
        "RAG 是什么？",
        max_turns=5,
        stream_callback=lambda e: None,  # 静默，避免刷屏
    )

    assert result.success is True
    assert "RAG" in result.content

    # 上下文里应记录了 search_documents 工具调用，且检索结果来自真实文档
    tool_messages = [
        m
        for m in result.context.messages
        if m.get("role") == "tool" and m.get("name") == "search_documents"
    ]
    assert len(tool_messages) == 1
    assert ("RAG" in tool_messages[0]["content"]) or (
        "检索" in tool_messages[0]["content"]
    )
