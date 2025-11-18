"""
Isolated test for arxiv_researcher_node
"""

import asyncio
import os
from textwrap import dedent

import arxiv
from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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


class ArxivFinding(BaseModel):
    title: str = Field(..., description="Title of the paper")
    authors: list[str] = Field(..., description="List of author names")
    summary: str = Field(..., description="Summary of the paper")
    url: str = Field(..., description="ArXiv URL")
    published: str = Field(..., description="Publication date")
    arxiv_id: str = Field(..., description="ArXiv identifier")


class ArxivFindings(BaseModel):
    papers: list[ArxivFinding] = Field(
        default_factory=list, description="List of found papers"
    )


class ResearchState(BaseModel):
    query: ResearchQuery
    plan: ResearchPlan | None = None
    arxiv_findings: list[ArxivFinding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    iteration: int = 0


@tool
def arxiv_search(query: str, max_results: int = 3) -> list[dict]:
    """Search for relevant papers using the arXiv tool"""
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        papers.append(
            {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "url": result.entry_id,
                "arxiv_id": result.entry_id.split("/")[-1],
            }
        )

    return papers


arxiv_parser = PydanticOutputParser(pydantic_object=ArxivFindings)

ARXIV_SEARCH_PROMPT = dedent("""
    You are an academic researcher specializing in analyzing arXiv papers.

    Your task is to:
    1. Search for relevant papers using the arXiv tool
    2. Extract key information from each paper
    3. Summarize the main contributions
    4. Identify the most important papers for the research topic

    Focus on recent, high-quality publications. Use the available tools to search arXiv.
""").strip()

ARXIV_PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
            You are a helpful assistant that extracts paper information from search results.
            {format_instructions}
            /no_think
        """).strip(),
        ),
        (
            "human",
            dedent("""
            Extract papers from these search results:

            {text}
        """).strip(),
        ),
    ]
)

arxiv_agent = create_agent(llm, [arxiv_search], system_prompt=ARXIV_SEARCH_PROMPT)


async def arxiv_researcher_node(state: ResearchState) -> dict:
    if not state.plan:
        return {"errors": ["No research plan available"]}

    # Use the agent to search for papers
    query = f"Search arXiv for: {state.plan.arxiv_query}. Find {state.query.max_papers} papers."
    result = await arxiv_agent.ainvoke({"messages": [HumanMessage(content=query)]})

    tool_messages = [msg.content for msg in result["messages"] if msg.type == "tool"]

    if tool_messages:
        # Create a chain with the parser
        prompt_with_parser = ARXIV_PARSER_PROMPT.partial(
            format_instructions=arxiv_parser.get_format_instructions()
        )
        chain = prompt_with_parser | llm | arxiv_parser

        try:
            # Invoke the chain with the tool results
            parsed_result = await chain.ainvoke({"text": str(tool_messages)})
            return {"arxiv_findings": parsed_result.papers}
        except Exception as e:
            return {
                "errors": [f"Failed to parse results: {str(e)}"],
                "arxiv_findings": [],
            }

    return {"arxiv_findings": []}


async def test_arxiv_researcher_node():
    query = ResearchQuery(
        topic="Machine learning optimization",
        max_papers=2,
        max_web_results=2,
    )

    plan = ResearchPlan(
        arxiv_query="machine learning optimization gradient descent",
        web_queries=["ML optimization"],
        focus_areas=["Algorithms"],
        expected_sources=4,
    )

    state = ResearchState(query=query, plan=plan)
    result = await arxiv_researcher_node(state)

    assert isinstance(result.get("arxiv_findings", []), list)
    findings = result.get("arxiv_findings", [])
    if len(findings) > 0:
        assert all(isinstance(f, ArxivFinding) for f in findings)
        assert all(f.title and f.authors and f.url for f in findings)

    print("Papers Found:", len(findings))
    for i, paper in enumerate(findings, 1):
        print(f"Paper {i}: {paper.title}")
        print(f"  Authors: {', '.join(paper.authors[:3])}")
        print(f"  URL: {paper.url}")
    print("PASSED")

    return result


if __name__ == "__main__":
    asyncio.run(test_arxiv_researcher_node())
