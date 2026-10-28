"""最小 MCP 客户端：演示并验证 MCP 的完整握手与工具调用流程。

它做四件事，正好对应 MCP 的三个协议环节：

1. ``initialize``               握手（协商版本、获取能力）
2. ``notifications/initialized`` 通知（按协议不需要响应）
3. ``tools/list``               发现服务端有哪些工具
4. ``tools/call``               调用 search_documents 做一次真实检索

用法::

    python -m rag_agent.mcp_demo_client "RAG 为什么要 overlap"

不传问题时使用默认问题。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

QUERY = "RAG 为什么要 overlap"

# 项目根目录（本文件在 rag_agent/ 下，取上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_requests(query: str) -> list[dict]:
    """按顺序构造 4 条消息（第 2 条是通知，无 id）。"""
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "demo-client", "version": "0.1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {"query": query, "top_k": 2},
            },
        },
    ]


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else QUERY
    requests = build_requests(query)
    payload = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in requests
    ) + "\n"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_agent.mcp_server"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    out, err = proc.communicate(payload, timeout=120)

    print("=" * 60)
    print("MCP 握手过程")
    print("=" * 60)
    if err.strip():
        print("[server stderr]", err.strip())

    responses = [json.loads(x) for x in out.splitlines() if x.strip()]
    print(f"\n收到 {len(responses)} 条响应（通知无响应，符合协议）")

    for resp in responses:
        rid = resp.get("id")
        result = resp.get("result", {})

        if rid == 1:
            print("\n[1] initialize ->")
            print(f"    protocolVersion = {result.get('protocolVersion')}")
            print(f"    capabilities    = {result.get('capabilities')}")
            print(f"    serverInfo      = {result.get('serverInfo')}")

        elif rid == 2:
            tools = result.get("tools", [])
            print(f"\n[2] tools/list -> 共 {len(tools)} 个工具")
            for t in tools:
                required = t["inputSchema"].get("required", [])
                print(
                    f"    - {t['name']}(required={required})"
                    f"：{t['description']}"
                )

        elif rid == 3:
            print("\n[3] tools/call search_documents ->")
            print(f"    isError = {result.get('isError')}")
            for item in result.get("content", []):
                text = item.get("text", "")
                print("    " + text.replace("\n", "\n    ")[:800])

    print("\n" + "=" * 60)
    print("MCP 全流程验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
