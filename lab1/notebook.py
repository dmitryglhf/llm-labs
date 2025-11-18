import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating tools
    """)
    return


@app.cell
def _():
    from langchain.tools import tool
    return (tool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Web search & extract
    """)
    return


@app.cell
def _():
    from ddgs import DDGS
    return (DDGS,)


@app.cell
def _(DDGS, tool):
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
        """
        DuckDuckGo text search for web pages, articles, and information.

        Args:
            query: text search query.
            region: us-en, uk-en, ru-ru, etc. Defaults to us-en.
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m, y. Defaults to None.
            max_results: maximum number of results. Defaults to 10.
            page: page of results. Defaults to 1.
            backend: A single or comma-delimited backends. Defaults to "auto".
        Returns:
            List of dictionaries with search results.
        """
        results = DDGS().text(
            query=query,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results,
            page=page,
            backend=backend,
        )
        return results
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arxiv search
    """)
    return


@app.cell
def _():
    import arxiv
    return (arxiv,)


@app.cell
def _(arxiv, tool):
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
    return (arxiv_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pydantic models and State
    """)
    return


@app.cell
def _():
    import operator
    from typing import Annotated

    from pydantic import BaseModel, Field
    return Annotated, BaseModel, Field, operator


@app.cell
def _(Annotated, BaseModel, Field, operator):
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
        plan: ResearchPlan | None = None
        arxiv_findings: Annotated[list[ArxivFinding], operator.add] = Field(
            default_factory=list
        )
        web_findings: Annotated[list[WebFinding], operator.add] = Field(
            default_factory=list
        )
        report: ResearchReport | None = None
        review: ReviewFeedback | None = None
        errors: Annotated[list[str], operator.add] = Field(default_factory=list)
        iteration: int = 0
    return (
        ArxivFinding,
        ResearchPlan,
        ResearchQuery,
        ResearchReport,
        ResearchState,
        ReviewFeedback,
        WebFinding,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph
    """)
    return


@app.cell
def _():
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
    from langgraph.types import RetryPolicy
    return (
        ChatOpenAI,
        END,
        HumanMessage,
        RetryPolicy,
        StateGraph,
        SystemMessage,
        create_agent,
    )


@app.cell
def _():
    import os

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    BASE_URL = os.getenv("OPENAI_BASE_URL", "http://a6k2.dgx:34000/v1")
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")
    return API_KEY, BASE_URL, MODEL_NAME


@app.cell
def _(API_KEY, BASE_URL, ChatOpenAI, MODEL_NAME):
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
    )
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prompts
    """)
    return


@app.cell
def _():
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
    return (
        ARXIV_RESEARCHER_PROMPT,
        PLANNER_PROMPT,
        REVIEWER_PROMPT,
        SYNTHESIZER_PROMPT,
        WEB_RESEARCHER_PROMPT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structure
    """)
    return


@app.cell
def _():
    import json
    return (json,)


@app.cell
def _(
    ARXIV_RESEARCHER_PROMPT,
    WEB_RESEARCHER_PROMPT,
    arxiv_search,
    create_agent,
    extract,
    llm,
    search,
):
    arxiv_agent = create_agent(
        llm, [arxiv_search], system_prompt=ARXIV_RESEARCHER_PROMPT
    )
    web_agent = create_agent(
        llm, [search, extract], system_prompt=WEB_RESEARCHER_PROMPT
    )
    return arxiv_agent, web_agent


@app.cell
def _(
    HumanMessage,
    PLANNER_PROMPT,
    ResearchPlan,
    ResearchState,
    SystemMessage,
    json,
    llm,
):
    async def planner_node(state: ResearchState) -> dict:
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=PLANNER_PROMPT
                    + f"\n\nYou must respond with valid JSON matching this schema: {ResearchPlan.model_json_schema()}"
                ),
                HumanMessage(
                    content=f"Topic: {state.query.topic}\nMax papers: {state.query.max_papers}\nMax web results: {state.query.max_web_results}"
                ),
            ]
        )

        parsed_content = json.loads(response.content)
        plan = ResearchPlan(**parsed_content)
        return {"plan": plan}
    return (planner_node,)


@app.cell
def _(ArxivFinding, HumanMessage, ResearchState, arxiv_agent, json, llm):
    async def arxiv_researcher_node(state: ResearchState) -> dict:
        if not state.plan:
            return {"errors": ["No research plan available"]}

        query = f"Search arXiv for: {state.plan.arxiv_query}. Find {state.query.max_papers} papers. Extract title, authors, summary, url, and published date for each paper."

        result = await arxiv_agent.ainvoke({"messages": [HumanMessage(content=query)]})

        ai_messages = [
            msg.content
            for msg in result["messages"]
            if msg.type == "tool" and msg.content
        ]

        if ai_messages:
            response = await llm.ainvoke(
                [
                    HumanMessage(
                        content=f"{ai_messages[-1]}\n\nExtract the findings as a JSON array matching this schema: {ArxivFinding.model_json_schema()}"
                    )
                ]
            )
            try:
                parsed_content = json.loads(response.content)
                arxiv_findings = [ArxivFinding(**item) for item in parsed_content]
                return {"arxiv_findings": arxiv_findings}
            except (json.JSONDecodeError, Exception):
                return {"arxiv_findings": []}

        return {"arxiv_findings": []}
    return (arxiv_researcher_node,)


@app.cell
def _(HumanMessage, ResearchState, WebFinding, json, llm, web_agent):
    async def web_researcher_node(state: ResearchState) -> dict:
        if not state.plan:
            return {"errors": ["No research plan available"]}

        queries = " AND ".join(state.plan.web_queries)
        query = f"Search web for: {queries}. Find {state.query.max_web_results} sources. Extract title, url, content summary, and assess relevance score for each source."

        result = await web_agent.ainvoke({"messages": [HumanMessage(content=query)]})

        ai_messages = [
            msg.content
            for msg in result["messages"]
            if msg.type == "ai" and msg.content
        ]

        if ai_messages:
            response = await llm.ainvoke(
                [
                    HumanMessage(
                        content=f"{ai_messages[-1]}\n\nExtract the findings as a JSON array matching this schema: {WebFinding.model_json_schema()}"
                    )
                ]
            )
            try:
                parsed_content = json.loads(response.content)
                web_findings = [WebFinding(**item) for item in parsed_content]
                return {"web_findings": web_findings}
            except (json.JSONDecodeError, Exception):
                return {"web_findings": []}

        return {"web_findings": []}
    return (web_researcher_node,)


@app.cell
def _(
    HumanMessage,
    ResearchReport,
    ResearchState,
    SYNTHESIZER_PROMPT,
    SystemMessage,
    json,
    llm,
):
    async def synthesizer_node(state: ResearchState) -> dict:
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
            feedback = "Previous review feedback: " + ", ".join(
                state.review.suggestions
            )

        context = f"""Topic: {state.query.topic}

    ArXiv Papers:
    {arxiv_summary if arxiv_summary else "No papers found"}

    Web Sources:
    {web_summary if web_summary else "No sources found"}

    {feedback}

    Respond with valid JSON matching this schema: {ResearchReport.model_json_schema()}"""

        response = await llm.ainvoke(
            [SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=context)]
        )

        try:
            parsed_content = json.loads(response.content)
            report = ResearchReport(**parsed_content)
            return {"report": report, "iteration": state.iteration + 1}
        except (json.JSONDecodeError, Exception) as e:
            return {
                "errors": [f"Failed to parse synthesizer response: {str(e)}"],
                "iteration": state.iteration + 1,
            }
    return (synthesizer_node,)


@app.cell
def _(
    HumanMessage,
    REVIEWER_PROMPT,
    ResearchState,
    ReviewFeedback,
    SystemMessage,
    json,
    llm,
):
    async def reviewer_node(state: ResearchState) -> dict:
        if not state.report:
            return {"errors": ["No report available to review"]}

        report_text = f"""Report Summary: {state.report.summary}
    Key Findings: {", ".join(state.report.key_findings)}
    Number of ArXiv papers: {len(state.report.arxiv_papers)}
    Number of Web sources: {len(state.report.web_sources)}
    Gaps identified: {", ".join(state.report.gaps_identified)}

    Respond with valid JSON matching this schema: {ReviewFeedback.model_json_schema()}"""

        response = await llm.ainvoke(
            [SystemMessage(content=REVIEWER_PROMPT), HumanMessage(content=report_text)]
        )

        try:
            parsed_content = json.loads(response.content)
            review = ReviewFeedback(**parsed_content)
            return {"review": review}
        except (json.JSONDecodeError, Exception) as e:
            return {"errors": [f"Failed to parse reviewer response: {str(e)}"]}
    return (reviewer_node,)


@app.cell
def _(ResearchState):
    def should_revise(state: ResearchState) -> str:
        if state.review and not state.review.approved and state.iteration < 2:
            return "synthesizer"
        return "end"
    return (should_revise,)


@app.cell
def _(ResearchState, StateGraph):
    g = StateGraph(ResearchState)
    return (g,)


@app.cell
def _(
    END,
    RetryPolicy,
    arxiv_researcher_node,
    g,
    planner_node,
    reviewer_node,
    should_revise,
    synthesizer_node,
    web_researcher_node,
):
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        backoff_factor=2.0,
        jitter=True,
    )

    g.add_node("planner", planner_node)
    g.add_node("arxiv_researcher", arxiv_researcher_node, retry=retry_policy)
    g.add_node("web_researcher", web_researcher_node, retry=retry_policy)
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

    app = g.compile()
    return (app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph Visualization
    """)
    return


@app.cell
def _(app, mo):
    mo.mermaid(app.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo Execution
    """)
    return


@app.cell
def _():
    QUERY = "Multi-agent systems with LLMs"
    return (QUERY,)


@app.cell
def _(QUERY, ResearchQuery):
    demo_query = ResearchQuery(
        topic=QUERY,
        max_papers=3,
        max_web_results=3,
    )
    return (demo_query,)


@app.cell
async def _(ResearchState, app, demo_query):
    initial_state = ResearchState(query=demo_query)
    result = await app.ainvoke(initial_state)
    return (result,)


@app.cell
def _(result):
    result
    return


if __name__ == "__main__":
    app.run()
