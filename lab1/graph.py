import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .models import (
    ArxivFinding,
    ResearchPlan,
    ResearchReport,
    ResearchState,
    ReviewFeedback,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
search_tool = DuckDuckGoSearchRun()

PLANNER_PROMPT = """You are a research coordinator. Your task is to analyze the research topic and create a structured research plan.

Based on the user's topic, generate:
1. A well-formulated arXiv query
2. Multiple web search queries to cover different aspects
3. Key focus areas for the research
4. Expected number of sources

Be specific and thorough in your planning."""

ARXIV_RESEARCHER_PROMPT = """You are an academic researcher specializing in analyzing arXiv papers.

Your task is to:
1. Search for relevant papers using the arXiv tool
2. Extract key information from each paper
3. Summarize the main contributions
4. Identify the most important papers for the research topic

Focus on recent, high-quality publications. Use the available tools to search arXiv."""

WEB_RESEARCHER_PROMPT = """You are a web researcher specializing in finding and analyzing online sources.

Your task is to:
1. Search the web using the DuckDuckGo tool
2. Extract and summarize relevant content
3. Assess the relevance and quality of each source
4. Provide concise summaries of key findings

Focus on authoritative and up-to-date sources. Use the available tools to search the web."""

SYNTHESIZER_PROMPT = """You are a research analyst and technical writer.

Your task is to synthesize all collected research into a comprehensive report.

You should:
1. Identify and summarize key findings across all sources
2. Write a coherent summary of the research landscape
3. Highlight important papers and sources
4. Identify gaps in current research or understanding

Create a well-structured, informative research report."""

REVIEWER_PROMPT = """You are a critical reviewer and research quality assessor.

Your task is to evaluate the research report for:
1. Completeness - are all important aspects covered?
2. Quality - is the analysis thorough and well-reasoned?
3. Clarity - is the report well-written and understandable?
4. Missing aspects - what could be improved or added?

Provide a quality score (0.0-1.0) and decide if the report is approved or needs revision.
Be strict but fair in your assessment."""


arxiv_agent = create_agent(llm, [arxiv_tool], system_prompt=ARXIV_RESEARCHER_PROMPT)
web_agent = create_agent(llm, [search_tool], system_prompt=WEB_RESEARCHER_PROMPT)


async def planner_node(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(ResearchPlan)

    response = await structured_llm.ainvoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(
                content=f"Topic: {state.query.topic}\nMax papers: {state.query.max_papers}\nMax web results: {state.query.max_web_results}"
            ),
        ]
    )

    state.plan = response
    return state


async def arxiv_researcher_node(state: ResearchState) -> ResearchState:
    if not state.plan:
        state.errors.append("No research plan available")
        return state

    query = f"Search arXiv for: {state.plan.arxiv_query}. Focus on: {', '.join(state.plan.focus_areas)}. Find {state.query.max_papers} papers. Extract title, authors, summary, url, and published date for each paper."

    try:
        result = await arxiv_agent.ainvoke({"messages": [HumanMessage(content=query)]})

        ai_messages = [
            msg.content
            for msg in result["messages"]
            if msg.type == "ai" and msg.content
        ]

        if ai_messages:
            structured_llm = llm.with_structured_output(list[ArxivFinding])
            findings = await structured_llm.ainvoke(
                [HumanMessage(content=ai_messages[-1])]
            )
            state.arxiv_findings = findings if findings else []
    except Exception as e:
        state.errors.append(f"ArXiv research error: {str(e)}")
        if state.retry_count < 3:
            state.retry_count += 1

    return state


async def web_researcher_node(state: ResearchState) -> ResearchState:
    if not state.plan:
        state.errors.append("No research plan available")
        return state

    queries = " AND ".join(state.plan.web_queries)
    query = f"Search web for: {queries}. Focus on: {', '.join(state.plan.focus_areas)}. Find {state.query.max_web_results} sources. Extract title, url, content summary, and assess relevance score for each source."

    try:
        result = await web_agent.ainvoke({"messages": [HumanMessage(content=query)]})

        ai_messages = [
            msg.content
            for msg in result["messages"]
            if msg.type == "ai" and msg.content
        ]

        if ai_messages:
            structured_llm = llm.with_structured_output(list[WebFinding])
            findings = await structured_llm.ainvoke(
                [HumanMessage(content=ai_messages[-1])]
            )
            state.web_findings = findings if findings else []
    except Exception as e:
        state.errors.append(f"Web research error: {str(e)}")
        if state.retry_count < 3:
            state.retry_count += 1

    return state


async def synthesizer_node(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(ResearchReport)

    arxiv_summary = "\n".join(
        [
            f"- {p.title} by {', '.join(p.authors)}: {p.summary}"
            for p in state.arxiv_findings
        ]
    )
    web_summary = "\n".join(
        [f"- {s.title} ({s.url}): {s.content_summary}" for s in state.web_findings]
    )

    feedback = ""
    if state.review and state.review.suggestions:
        feedback = "Previous review feedback: " + ", ".join(state.review.suggestions)

    context = f"""Topic: {state.query.topic}

ArXiv Papers:
{arxiv_summary if arxiv_summary else "No papers found"}

Web Sources:
{web_summary if web_summary else "No sources found"}

{feedback}"""

    response = await structured_llm.ainvoke(
        [SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=context)]
    )

    state.report = response
    state.iteration += 1
    return state


async def reviewer_node(state: ResearchState) -> ResearchState:
    if not state.report:
        state.errors.append("No report available to review")
        return state

    structured_llm = llm.with_structured_output(ReviewFeedback)

    report_text = f"""Report Summary: {state.report.summary}
Key Findings: {", ".join(state.report.key_findings)}
Number of ArXiv papers: {len(state.report.arxiv_papers)}
Number of Web sources: {len(state.report.web_sources)}
Gaps identified: {", ".join(state.report.gaps_identified)}"""

    response = await structured_llm.ainvoke(
        [SystemMessage(content=REVIEWER_PROMPT), HumanMessage(content=report_text)]
    )

    state.review = response
    return state


def should_revise(state: ResearchState) -> Literal["synthesizer", "end"]:
    if state.review and not state.review.approved and state.iteration < 2:
        return "synthesizer"
    return "end"


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("planner", planner_node)
    g.add_node("arxiv_researcher", arxiv_researcher_node)
    g.add_node("web_researcher", web_researcher_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("reviewer", reviewer_node)

    g.set_entry_point("planner")

    g.add_edge("planner", "arxiv_researcher")
    g.add_edge("planner", "web_researcher")

    g.add_edge("arxiv_researcher", "synthesizer")
    g.add_edge("web_researcher", "synthesizer")

    g.add_edge("synthesizer", "reviewer")

    g.add_conditional_edges(
        "reviewer", should_revise, {"synthesizer": "synthesizer", "end": END}
    )

    return g.compile()


if __name__ == "__main__":
    app = build_graph()

    mermaid_png = app.get_graph().draw_mermaid_png()
    with open("research_graph.png", "wb") as f:
        f.write(mermaid_png)

    print("Graph visualization saved to research_graph.png")

    mermaid_code = app.get_graph().draw_mermaid()
    print("\nMermaid code:")
    print(mermaid_code)
