from typing import Optional

from pydantic import BaseModel


class ResearchQuery(BaseModel):
    topic: str
    max_papers: int = 5
    max_web_results: int = 5
    year_filter: Optional[str] = None


class ResearchPlan(BaseModel):
    arxiv_query: str
    web_queries: list[str]
    focus_areas: list[str]
    expected_sources: int


class ArxivFinding(BaseModel):
    title: str
    authors: list[str]
    summary: str
    url: str
    published: str


class WebFinding(BaseModel):
    title: str
    url: str
    content_summary: str
    relevance_score: float


class ResearchFindings(BaseModel):
    arxiv_papers: list[ArxivFinding]
    web_sources: list[WebFinding]
    total_sources: int


class ResearchReport(BaseModel):
    topic: str
    key_findings: list[str]
    summary: str
    arxiv_papers: list[ArxivFinding]
    web_sources: list[WebFinding]
    gaps_identified: list[str]


class ReviewFeedback(BaseModel):
    approved: bool
    missing_aspects: list[str]
    quality_score: float
    suggestions: list[str]


class ResearchState(BaseModel):
    query: ResearchQuery
    plan: Optional[ResearchPlan] = None
    arxiv_findings: list[ArxivFinding] = []
    web_findings: list[WebFinding] = []
    report: Optional[ResearchReport] = None
    review: Optional[ReviewFeedback] = None
    retry_count: int = 0
    errors: list[str] = []
    iteration: int = 0
