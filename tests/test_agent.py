"""agent 组装测试（不触发真实 LLM 调用）。"""

from rag_agent.agent import create_rag_agent
from rag_agent.tools import KnowledgeBase


def test_create_rag_agent():
    kb = KnowledgeBase()
    agent = create_rag_agent(kb, api_key="test-key")
    assert agent.name == "DocRAGAgent"
    assert len(agent.tools) == 3
    # 系统消息里应包含检索约束
    assert "search_documents" in agent.get_system_message()["content"]
