"""MCP Server：把 doc-rag-agent 的文档检索能力通过 MCP 协议暴露出去。

为什么要有这个文件
------------------
MCP（Model Context Protocol）本质是「标准化的工具接入协议」：任何支持
MCP 的 host（Cursor、Claude Desktop 等）都能通过同一套 JSON-RPC 接口
发现并调用本项目的知识库检索能力，而不需要为每个 host 单独适配。

传输层是 stdio，协议是 JSON-RPC 2.0：每条消息一行 JSON（UTF-8），
依次完成三个环节：

1. ``initialize``  握手，返回协议版本、能力集、服务端信息
2. ``tools/list``  声明本服务有哪些工具（名字 + 描述 + JSON Schema）
3. ``tools/call``  执行某个工具，返回内容

这里只用 Python 标准库实现，不依赖官方 mcp SDK，便于看清协议细节，
也保证在离线 / 无外网环境下可以直接运行。

用法
----
作为 MCP Server 启动（供 host 接入）::

    python -m rag_agent.mcp_server

带一个最小客户端跑通全流程（可直接看到效果）::

    python -m rag_agent.mcp_demo_client "RAG 为什么要 overlap"
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any

from .tools import KnowledgeBase

# 协议版本：客户端会带自己的版本，服务端回显以完成协商
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "doc-rag-agent"
SERVER_VERSION = "0.1.0"

# data/ 目录（相对项目根目录），用于启动时自动加载
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# 工具声明：MCP 用这份 JSON Schema 告诉 host「我有哪些能力、参数怎么填」
TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": (
            "在已加载的文档知识库中检索与问题最相关的片段"
            "（基于 BM25 关键词检索，中文分词使用 jieba）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的问题或关键词",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的片段数量，默认 3",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "load_documents",
        "description": "加载本地文档到知识库，多个路径用英文逗号分隔。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "string",
                    "description": (
                        "文档路径，多个用英文逗号分隔，"
                        '例如 "data/a.md,data/b.txt"'
                    ),
                }
            },
            "required": ["paths"],
        },
    },
    {
        "name": "list_documents",
        "description": "列出知识库中已加载的文档及各文档片段数量。",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


class DocRagMCPServer:
    """MCP Server 的协议实现。

    与传输层解耦：``handle`` 接收一条已解析的 JSON-RPC 消息并返回响应
    字典（通知类消息返回 None），这样单测可以直接在进程内驱动它，
    不必起子进程。
    """

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self.kb = kb if kb is not None else KnowledgeBase()

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """执行工具并返回纯文本结果（MCP 的 content 项就是文本）。"""
        if name == "search_documents":
            chunks = self.kb.search(
                str(arguments["query"]), int(arguments.get("top_k", 3))
            )
            if not chunks:
                return (
                    "知识库为空或没有检索到相关内容，"
                    "请先用 load_documents 加载文档。"
                )
            return "\n\n---\n\n".join(
                f"[来源: {c.source} 片段#{c.index}]\n{c.text}"
                for c in chunks
            )

        if name == "load_documents":
            paths = [
                p.strip()
                for p in str(arguments["paths"]).split(",")
                if p.strip()
            ]
            if not paths:
                return "未提供有效的文件路径。"
            count = self.kb.load(paths)
            return f"成功加载 {len(paths)} 个文件，新增 {count} 个片段。"

        if name == "list_documents":
            if not self.kb.chunks:
                return "知识库为空。"
            counts = Counter(c.source for c in self.kb.chunks)
            lines = [f"- {src}: {n} 个片段" for src, n in counts.items()]
            return "知识库内容:\n" + "\n".join(lines)

        raise ValueError(f"未知工具：{name}")

    # ------------------------------------------------------------------
    # JSON-RPC 分发
    # ------------------------------------------------------------------
    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """处理一条 JSON-RPC 消息，返回响应；通知类消息返回 None。"""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # 通知（notifications/*）没有 id，按协议不返回任何内容
        if msg_id is None:
            return None

        try:
            if method == "initialize":
                return self._ok(
                    msg_id,
                    {
                        # 回显客户端请求的版本，完成版本协商
                        "protocolVersion": params.get(
                            "protocolVersion", PROTOCOL_VERSION
                        ),
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": SERVER_NAME,
                            "version": SERVER_VERSION,
                        },
                    },
                )

            if method == "ping":
                return self._ok(msg_id, {})

            if method == "tools/list":
                return self._ok(msg_id, {"tools": TOOLS})

            if method == "tools/call":
                name = params.get("name")
                if not name:
                    return self._err(msg_id, -32602, "缺少参数 name")
                arguments = params.get("arguments") or {}
                text = self.call_tool(str(name), arguments)
                return self._ok(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                )

            return self._err(msg_id, -32601, f"未知方法：{method}")

        except Exception as exc:  # noqa: BLE001 - 工具异常要转成 isError 结果
            if method == "tools/call":
                return self._ok(
                    msg_id,
                    {
                        "content": [
                            {"type": "text", "text": f"工具执行失败：{exc}"}
                        ],
                        "isError": True,
                    },
                )
            return self._err(msg_id, -32603, str(exc))

    @staticmethod
    def _ok(msg_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }


def autoload(server: DocRagMCPServer) -> int:
    """启动时自动加载 data/ 下的文档，让 server 开箱即用。"""
    if not os.path.isdir(DATA_DIR):
        return 0
    paths = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith((".md", ".txt"))
    )
    if not paths:
        return 0
    return server.kb.load(paths)


def main() -> None:  # pragma: no cover - 由 tests/test_mcp_server.py 的子进程测试覆盖
    """stdio 主循环：读一行 JSON → 处理 → 写一行 JSON。"""
    # stdout 是协议通道，只允许输出协议 JSON；日志一律走 stderr
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    server = DocRagMCPServer()
    loaded = autoload(server)
    print(
        f"[mcp_server] {SERVER_NAME} v{SERVER_VERSION} ready "
        f"(chunks={len(server.kb.chunks)}, autoload={loaded})",
        file=sys.stderr,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[mcp_server] 无法解析的消息: {exc}", file=sys.stderr)
            continue

        response = server.handle(message)
        if response is not None:
            sys.stdout.write(
                json.dumps(response, ensure_ascii=False) + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
