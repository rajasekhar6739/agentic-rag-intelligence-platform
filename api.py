from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents.tool_agent import ToolAgent


app = FastAPI(
    title="Agentic RAG AI",
    description="Agentic AI system with RAG, tool calling and evaluation.",
    version="1.0.0",
)

agent = ToolAgent()


class Document(BaseModel):
    text: str
    source: str = "unknown"
    page: Optional[int] = None
    chunk_id: Optional[int] = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    success: bool
    answer: str
    grounded: bool
    confidence: float
    citations: List[str]
    tools_used: List[str]
    iterations: int
    evidence: List[Any]


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Agentic RAG AI",
        "status": "running",
        "tools": agent.list_tools(),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "tools": agent.list_tools(),
    }


@app.post("/documents")
def load_documents(
    documents: List[Document],
) -> Dict[str, Any]:

    result = agent.load_documents(
        [doc.model_dump() for doc in documents]
    )

    return {
        "success": True,
        "result": result,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
) -> QueryResponse:

    result = agent.run(
        request.question,
        top_k=request.top_k,
    )

    return QueryResponse(
        success=result.get(
            "success",
            False,
        ),
        answer=result.get(
            "answer",
            "",
        ),
        grounded=result.get(
            "grounded",
            False,
        ),
        confidence=result.get(
            "confidence",
            0.0,
        ),
        citations=result.get(
            "citations",
            [],
        ),
        tools_used=result.get(
            "tools_used",
            [],
        ),
        iterations=result.get(
            "iterations",
            0,
        ),
        evidence=result.get(
            "evidence",
            [],
        ),
    )