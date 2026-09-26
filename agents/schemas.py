from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    text: str
    source: str
    page: int
    chunk_id: int
    token_count: int = 0


class Evidence(BaseModel):
    source: str
    page: int
    quote: str
    relevance: float = Field(ge=0.0, le=1.0)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: dict


class AnalysisResult(BaseModel):
    summary: str
    root_causes: list[str]
    evidence: list[Evidence]
    recommended_actions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool

class SourceEvidence(BaseModel):
    source: str
    page: int
    chunk_id: int
    text: str
    score: float


class AgentAnswer(BaseModel):
    answer: str
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )
    sources: list[SourceEvidence] = []
    tools_used: list[str] = []