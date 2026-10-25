# doc-rag-agent

基于 RAG（检索增强生成）的本地文档知识库问答 Agent。加载本地 Markdown / 文本文件，自动切分、检索相关片段，再交给大模型生成有据可依的回答。

## ✨ 特性

- 🧠 **RAG 全链路**：文档加载 → 切片 → BM25 检索 → LLM 生成回答
- 🔧 **工具调用**：检索/加载/列表三种能力以 function calling 工具形式暴露给 LLM
- 💬 **多轮对话**：基于上下文管理，支持连续追问
- 🌊 **流式输出**：实时展示 Agent 的思考与工具调用过程
- 🇨🇳 **中文友好**：jieba 分词，中文文档效果好
- ⚡ **零外部检索依赖**：BM25 纯本地运行，无需 embedding 服务

## 🏗️ 架构

```
用户提问
   │
   ▼
┌─────────────┐    function calling    ┌──────────────────┐
│  DocRAGAgent │ ────────────────────► │ search_documents  │
│  (DeepSeek)  │ ◄──────────────────── │  (BM25 检索工具)   │
└─────────────┘      返回相关片段       └──────────────────┘
   │                                          ▲
   │ 生成回答                                   │
   ▼                                          │
  回答（引用来源）                        ┌──────┴──────┐
                                      │  知识库       │
                                      │ (切分后的片段) │
                                      └─────────────┘
```

数据流：

1. `chunking.py` 把文档按段落切分，长段落用滑动窗口切片（带 overlap）
2. `retriever.py` 用 jieba 分词 + BM25 建索引并检索
3. `tools.py` 用 `@function_tool` 把「检索/加载/列表」封装成 Agent 工具
4. `agent.py` 组装 DeepSeek 模型 + 工具，约束它「先检索再回答」
5. `cli.py` 提供交互式入口

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 DeepSeek API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 key：
# API_KEY=sk-xxxxxxxx
```

### 3. 放入你的文档

把 `.md` / `.txt` 文件丢进 `data/` 目录（已内置一篇示例）。

### 4. 无 Key 快速体验检索（可选）

还没配 key 时，可以先直观看到「切分 → 检索」效果：

```bash
python -m rag_agent.demo
```

### 5. 运行（需要 key）

```bash
python -m rag_agent.cli
```

```
已自动加载 1 个文件，共 5 个片段。
==================================================
📚 文档知识库问答 Agent（RAG）
输入问题开始提问，输入 quit / exit 退出
==================================================

你: DeepSeek 的 API 价格是多少？
📝 问题：DeepSeek 的 API 价格是多少？
🔧 工具：search_documents(...)
📊 工具结果：[来源: data/sample.md 片段#1] ...
✅ 回答：DeepSeek 的输入价格为每百万 token 约 1 元人民币...
```

### 6. Web 界面（可视化操作）

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501，可上传文档、实时查看 Agent 的「思考 → 工具调用 → 检索片段 → 回答」全过程。

## 🧪 测试

```bash
python -m pytest
```

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| Agent 框架 | [ZipAgent](https://github.com/JiayuXu0/ZipAgent)（复用 Agent / Runner / Tool 抽象） |
| LLM | DeepSeek `deepseek-chat`（OpenAI 兼容） |
| 检索 | BM25（jieba + rank_bm25） |
| 切分 | 段落切分 + 滑动窗口（overlap） |
| 语言 | Python 3.10+ |

## 📂 项目结构

```
doc-rag-agent/
├── rag_agent/
│   ├── chunking.py    # 文档切分
│   ├── retriever.py   # BM25 检索
│   ├── tools.py       # function calling 工具封装
│   ├── agent.py       # 组装 RAG Agent
│   └── cli.py         # 交互式入口
├── data/              # 你的知识库文档
├── tests/             # 单元测试
└── README.md
```

## 🗺️ 后续规划

- [ ] 向量检索（embedding + 余弦相似度 / FAISS），与 BM25 做混合召回
- [ ] 支持 PDF 文档（pypdf）
- [ ] Streamlit Web 界面
- [ ] 会话持久化与多知识库

## 📄 许可证

MIT
