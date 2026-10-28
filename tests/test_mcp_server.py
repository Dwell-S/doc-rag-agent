"""mcp_server 测试：验证 MCP 协议实现是否正确。

分两层：
1. 进程内测试：直接驱动 ``DocRagMCPServer.handle``，覆盖各分支；
2. 子进程测试：通过真实 stdio 管道跑通 initialize → tools/list →
   tools/call，确认协议在真实传输层可用。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from rag_agent.mcp_server import (
    PROTOCOL_VERSION,
    DocRagMCPServer,
    autoload,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = "data/sample.md"


def _req(msg_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _text(result):
    """从 tools/call 结果里取出文本内容。"""
    return "\n".join(item.get("text", "") for item in result["content"])


# ----------------------------------------------------------------------
# 1. 进程内协议测试
# ----------------------------------------------------------------------
def test_initialize_returns_negotiated_capabilities():
    server = DocRagMCPServer()
    resp = server.handle(
        _req(
            1,
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}},
        )
    )
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    # 版本协商：回显客户端请求的版本
    assert result["protocolVersion"] == "2024-11-05"
    # 能力集：声明支持 tools
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "doc-rag-agent"


def test_notification_has_no_response():
    server = DocRagMCPServer()
    # notifications/initialized 无 id，按协议不返回任何内容
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_declares_schema():
    server = DocRagMCPServer()
    resp = server.handle(_req(2, "tools/list"))
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"search_documents", "load_documents", "list_documents"}

    search = next(t for t in tools if t["name"] == "search_documents")
    schema = search["inputSchema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["properties"]["top_k"]["type"] == "integer"


def test_call_load_then_search_and_list():
    server = DocRagMCPServer()

    # 未加载时检索：给出引导语而不是报错
    empty = server.handle(
        _req(3, "tools/call", {"name": "search_documents", "arguments": {"query": "RAG"}})
    )
    assert empty["result"]["isError"] is False
    assert "加载文档" in _text(empty["result"])

    # 加载文档
    loaded = server.handle(
        _req(4, "tools/call", {"name": "load_documents", "arguments": {"paths": SAMPLE}})
    )
    assert loaded["result"]["isError"] is False
    assert "片段" in _text(loaded["result"])

    # 检索
    searched = server.handle(
        _req(
            5,
            "tools/call",
            {"name": "search_documents", "arguments": {"query": "RAG 什么是", "top_k": 2}},
        )
    )
    assert searched["result"]["isError"] is False
    text = _text(searched["result"])
    assert "来源" in text

    # 列表
    listed = server.handle(_req(6, "tools/call", {"name": "list_documents", "arguments": {}}))
    assert "data/sample.md" in _text(listed["result"])


def test_unknown_method_and_bad_params():
    server = DocRagMCPServer()

    unknown = server.handle(_req(7, "no/such/method"))
    assert unknown["error"]["code"] == -32601

    missing_name = server.handle(_req(8, "tools/call", {"arguments": {}}))
    assert missing_name["error"]["code"] == -32602


def test_unknown_tool_returns_is_error():
    server = DocRagMCPServer()
    resp = server.handle(
        _req(9, "tools/call", {"name": "not_a_tool", "arguments": {}})
    )
    # 工具级异常转成 isError 结果，而不是 JSON-RPC 错误
    assert resp["result"]["isError"] is True
    assert "未知工具" in _text(resp["result"])


def test_autoload_fills_knowledge_base():
    server = DocRagMCPServer()
    autoload(server)
    assert len(server.kb.chunks) > 0


def test_ping_and_default_protocol_version():
    server = DocRagMCPServer()

    # ping 是 MCP 客户端的保活检查，返回空结果
    assert server.handle(_req(11, "ping"))["result"] == {}

    # 客户端未指定版本时，回退到服务端默认版本
    resp = server.handle(_req(12, "initialize", {"capabilities": {}}))
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_empty_inputs_return_hints():
    server = DocRagMCPServer()

    # 空路径 → 明确提示，而不是抛异常
    loaded = server.handle(
        _req(
            13,
            "tools/call",
            {"name": "load_documents", "arguments": {"paths": " , , "}},
        )
    )
    assert loaded["result"]["isError"] is False
    assert "未提供有效的文件路径" in _text(loaded["result"])

    # 空知识库 → 友好提示
    listed = server.handle(
        _req(14, "tools/call", {"name": "list_documents", "arguments": {}})
    )
    assert "知识库为空" in _text(listed["result"])


def test_missing_required_argument_marks_error():
    server = DocRagMCPServer()
    # 缺少必填参数 query：工具内部抛 KeyError，应转成 isError 结果
    resp = server.handle(
        _req(15, "tools/call", {"name": "search_documents", "arguments": {}})
    )
    assert resp["result"]["isError"] is True


# ----------------------------------------------------------------------
# 2. 真实 stdio 传输测试
# ----------------------------------------------------------------------
def test_stdio_transport_round_trip():
    """通过子进程管道跑完整握手，验证 stdio 传输层真实可用。"""
    requests = [
        _req(
            1,
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}},
        ),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _req(2, "tools/list"),
        _req(
            3,
            "tools/call",
            {
                "name": "search_documents",
                "arguments": {"query": "overlap", "top_k": 1},
            },
        ),
    ]
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"

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

    assert proc.returncode == 0, f"server 异常退出: {err}"

    lines = [x for x in out.splitlines() if x.strip()]
    # 通知无响应，因此 3 条请求 → 3 条响应
    assert len(lines) == 3, f"响应条数不对: {out!r}"

    responses = {json.loads(x)["id"]: json.loads(x) for x in lines}

    assert responses[1]["result"]["serverInfo"]["name"] == "doc-rag-agent"
    assert len(responses[2]["result"]["tools"]) == 3

    call_result = responses[3]["result"]
    assert call_result["isError"] is False
    assert call_result["content"][0]["type"] == "text"
    assert call_result["content"][0]["text"].strip()


@pytest.mark.parametrize("query", ["RAG", "overlap", "BM25"])
def test_search_returns_content_for_various_queries(query):
    """检索结果对多个关键词都非空（知识库已自动加载）。"""
    server = DocRagMCPServer()
    autoload(server)
    resp = server.handle(
        _req(
            10,
            "tools/call",
            {"name": "search_documents", "arguments": {"query": query}},
        )
    )
    assert resp["result"]["isError"] is False
    assert _text(resp["result"]).strip()
