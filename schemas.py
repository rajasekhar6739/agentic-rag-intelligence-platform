from pydantic import BaseModel, Field

class Evidence(BaseModel):
    source: str
    page: int
    quote: str
    relevance: float = Field(ge=0, le=1)

class AnalysisResult(BaseModel):
    summary: str
    root_causes: list[str]
    evidence: list[Evidence]
    recommended_actions: list[str]
    confidence: float = Field(ge=0, le=1)
    requires_human_review: bool
