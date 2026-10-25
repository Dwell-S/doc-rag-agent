"""组装 RAG Agent：把 DeepSeek 模型 + 知识库工具 组合成一个 Agent。"""

from zipagent import Agent, OpenAIModel

from .tools import KnowledgeBase, build_tools


def create_rag_agent(
    kb: KnowledgeBase,
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
) -> Agent:
    """创建文档知识库问答 Agent。

    api_key 传 None 时会从环境变量 API_KEY 读取（OpenAIModel 的默认行为），
    因此生产用法是：在 .env 里填 MODEL / API_KEY / BASE_URL，代码里不硬编码。
    """
    llm = OpenAIModel(model=model, api_key=api_key, base_url=base_url)

    instructions = (
        "你是一个基于文档知识库的问答助手。\n"
        "回答用户问题时，必须先用 search_documents 工具检索知识库，"
        "再根据检索到的片段作答，不要凭空编造。\n"
        "如果检索结果为空，如实告诉用户知识库里没有相关信息。\n"
        "回答时尽量说明信息来源。"
    )

    return Agent(
        name="DocRAGAgent",
        instructions=instructions,
        model=llm,
        tools=build_tools(kb),
    )
