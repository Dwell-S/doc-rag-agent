# DeepSeek 简介

DeepSeek 是一家中国的人工智能公司，专注于大语言模型的研发。DeepSeek 推出的 DeepSeek-V3 和 DeepSeek-R1 模型在数学推理、代码生成等任务上表现出色，凭借出色的性价比在开发者社区中广受欢迎。

## DeepSeek 的 API

DeepSeek 提供 OpenAI 兼容的 API 接口，开发者可以把 base_url 设置为 https://api.deepseek.com，并使用 deepseek-chat 模型进行对话。它的价格远低于同类产品，输入价格为每百万 token 约 1 元人民币，缓存命中时价格更低，因此非常适合个人开发者和小团队使用。

## 为什么适合做 Agent

DeepSeek 的模型支持 function calling，也就是工具调用能力。这让它非常适合用来构建 AI Agent：你可以让模型在需要时调用外部工具，例如检索文档、查询天气、执行计算、读写文件等。工具调用是 agent 区别于普通聊天机器人的核心能力。

## RAG 与知识库问答

检索增强生成（RAG，Retrieval-Augmented Generation）是一种让大模型回答特定领域问题的方法。它的基本流程是：先把本地文档切分成若干片段，用检索算法找出与问题最相关的片段，再把这些片段作为上下文交给模型生成答案。这样能显著减少模型"胡编乱造"（幻觉）的概率，并让模型能够回答训练数据里没有的私有知识。

## 小结

把 DeepSeek 这样的便宜模型、function calling 工具调用、以及 RAG 检索三者结合起来，就可以构建一个成本低、效果好、可私有化部署的文档知识库问答系统。这也是当前 agent 应用开发方向最常见的工程实践。
