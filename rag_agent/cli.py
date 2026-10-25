"""命令行交互入口。

用法（在项目根目录 doc-rag-agent 下）:
    .venv\\Scripts\\python -m rag_agent.cli

会自动加载 data/ 目录下的 .md/.txt 文档，然后进入交互式问答。
"""

import os
import sys

from dotenv import load_dotenv
from zipagent import Runner

from .agent import create_rag_agent
from .tools import KnowledgeBase

load_dotenv()


def main() -> None:
    # Windows 控制台默认 GBK 编码，重配为 UTF-8 以正确打印 emoji
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # 检查 API Key 是否已配置，避免直接抛 traceback
    if not os.getenv("API_KEY"):
        print("❌ 未检测到 API_KEY。")
        print("请先复制 .env.example 为 .env，并在其中填入你的 DeepSeek API Key：")
        print("   copy .env.example .env")
        print("   # 编辑 .env，设置 API_KEY=sk-你的key")
        return

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
