# AI Agentic RAG System

An end-to-end Agentic AI system combining RAG, hybrid retrieval, reranking,
tool calling, structured execution, multimodal AI, evaluation, and
self-correction.

## Core Capabilities

- LLM-based agent planning
- Agentic workflows
- Retrieval-Augmented Generation (RAG)
- Semantic vector retrieval
- BM25 lexical retrieval
- Hybrid retrieval
- Reranking
- Tool/function calling
- Calculator tool execution
- Structured tool outputs
- Evidence-grounded answers
- Citation tracking
- Critic-based verification
- Retry/self-correction workflow
- Multimodal image analysis
- Agent evaluation metrics
- End-to-end evaluation

## Architecture

```text
User Query
    |
    v
Agent Orchestrator
    |
    v
Planner
    |
    +--------------------+
    |                    |
    v                    v
RAG Search          Tool Calling
    |                    |
    v                    v
Vector Search       Calculator / Tools
    |
    v
BM25 Retrieval
    |
    v
Hybrid Ranking
    |
    v
Reranker
    |
    v
Evidence
    |
    v
Answer Generation
    |
    v
Critic / Evaluator
    |
    +---- supported ----> Final Answer
    |
    +---- unsupported --> Retry / Correction