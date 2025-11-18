"""
Isolated test for web_researcher_node
"""

import asyncio
import json
import os

import requests
from ddgs import DDGS
from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from markitdown import MarkItDown
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(usecwd=True))
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME)  # type: ignore


class ResearchQuery(BaseModel):
    topic: str
    max_papers: int = 5
    max_web_results: int = 5
    year_filter: str | None = None


class ResearchPlan(BaseModel):
    arxiv_query: str
    web_queries: list[str]
    focus_areas: list[str]
    expected_sources: int


class WebFinding(BaseModel):
    title: str
    url: str
    content_summary: str
    relevance_score: float


class ResearchState(BaseModel):
    query: ResearchQuery
    plan: ResearchPlan | None = None
    web_findings: list[WebFinding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    iteration: int = 0


DDGSResult = list[dict[str, str]]


@tool
def search(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    page: int = 1,
    backend: str = "auto",
) -> DDGSResult | str:
    """Searches for information related to the given query using DuckDuckGo."""
    results = DDGS().text(  # type: ignore
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
        page=page,
        backend=backend,
    )
    return results


def truncate_text(text_content: str, max_lines: int | None = None) -> str:
    if max_lines is None:
        return text_content
    lines = text_content.split("\n")
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines])
        return (
            f"{truncated}\n\n... (truncated, showing {max_lines} of {len(lines)} lines)"
        )
    return text_content


@tool
async def extract(url: str, max_lines: int | None = None) -> str:
    """Extracts and summarizes relevant content from a given URL."""

    def _convert():
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
        )
        md = MarkItDown(requests_session=session)
        result = md.convert(str(url))
        return result.text_content

    markdown_content = await asyncio.to_thread(_convert)
    markdown_content = truncate_text(markdown_content, max_lines)
    return markdown_content


WEB_RESEARCHER_PROMPT = """You are a web researcher specializing in finding and analyzing online sources.

Your task is to:
1. Search the web using the DuckDuckGo tool
2. Extract and summarize relevant content
3. Assess the relevance and quality of each source
4. Provide concise summaries of key findings

Focus on authoritative and up-to-date sources. Use the available tools to search the web."""

web_agent = create_agent(llm, [search, extract], system_prompt=WEB_RESEARCHER_PROMPT)


async def web_researcher_node(state: ResearchState) -> dict:
    if not state.plan:
        return {"errors": ["No research plan available"]}

    queries = " AND ".join(state.plan.web_queries)
    query = f"Search web for: {queries}. Find {state.query.max_web_results} sources. Extract title, url, content summary, and assess relevance score for each source."

    result = await web_agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=query),
            ]
        }
    )

    if result:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=f"{result['messages']}\n\nExtract the findings as a JSON array matching this schema: {WebFinding.model_json_schema()}"
                )
            ]
        )
        try:
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            parsed_content = json.loads(content)
            web_findings = [WebFinding(**item) for item in parsed_content]
            print("Error")
            return {"web_findings": web_findings}
        except (json.JSONDecodeError, Exception):
            print("Error")
            return {"web_findings": []}

    print("Error")
    return {"web_findings": []}


async def test_web_researcher_node():
    query = ResearchQuery(
        topic="Python programming tutorials. Extract and summarize information from first= link.",
        max_papers=2,
        max_web_results=3,
    )

    plan = ResearchPlan(
        arxiv_query="python programming",
        web_queries=["Python tutorial", "Python beginner guide"],
        focus_areas=["Basics", "Syntax"],
        expected_sources=5,
    )

    state = ResearchState(query=query, plan=plan)
    result = await web_researcher_node(state)

    assert isinstance(result.get("web_findings", []), list)
    findings = result.get("web_findings", [])

    print("Sources Found:", len(findings))
    for i, source in enumerate(findings, 1):
        print(f"Source {i}: {source.title}")
        print(f"  URL: {source.url}")
        print(f"  Relevance: {source.relevance_score}")
        print(f"  Content Summary: {source.content_summary}")
    print("PASSED")

    return result


if __name__ == "__main__":
    asyncio.run(test_web_researcher_node())
