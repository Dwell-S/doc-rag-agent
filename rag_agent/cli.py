"""命令行交互入口。

用法（在项目根目录 doc-rag-agent 下）:
    .venv\\Scripts\\python -m rag_agent.cli

会自动加载 data/ 目录下的 .md/.txt 文档，然后进入交互式问答。
"""

import os

from dotenv import load_dotenv
from zipagent import Runner

from .agent import create_rag_agent
from .tools import KnowledgeBase

load_dotenv()


def main() -> None:
    kb = KnowledgeBase()

    # 自动加载 data/ 目录下的文档
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.isdir(data_dir):
        paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith((".md", ".txt"))
        ]
        if paths:
            n = kb.load(paths)
            print(f"已自动加载 {len(paths)} 个文件，共 {n} 个片段。")

    agent = create_rag_agent(kb)

    print("=" * 50)
    print("📚 文档知识库问答 Agent（RAG）")
    print("输入问题开始提问，输入 quit / exit 退出")
    print("=" * 50)

    context = None
    while True:
        try:
            q = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if q.lower() in ("quit", "exit", "q", "退出"):
            break
        if not q:
            continue

        # Runner.run 会在控制台打印 思考/工具调用/最终回答 的完整过程
        result = Runner.run(agent, q, context=context)
        context = result.context  # 保留多轮对话上下文


if __name__ == "__main__":
    main()
