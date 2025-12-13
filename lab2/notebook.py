import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 2: Multi-Agent Study/Productivity Assistant

    This notebook implements a multi-agent system using LangChain and LangGraph to assist with study, programming, and productivity tasks.

    ## System Architecture

    The system consists of 5 specialized agents:
    - **Router Agent**: Classifies requests and determines which agent to activate (conditional routing)
    - **Theory Agent**: Handles conceptual/theoretical questions about MAS and LLMs
    - **Code Agent**: Assists with programming and implementation questions (uses code execution tool)
    - **Planning Agent**: Helps with task planning and productivity (uses time and planning tools)
    - **Memory Agent**: Manages session history and user context, saves memory to file

    ### MAS Pattern: Router + Specialized Agents with Conditional Routing

    The system uses a **router pattern** where the Router Agent analyzes incoming queries and routes them to the appropriate specialized agent using **conditional edges**. Only ONE agent is activated per query based on classification.

    ### Flow Diagram

    ```mermaid
    graph TD
        A[User Query] --> B[Router Agent]
        B -->|theory| C[Theory Agent]
        B -->|code| D[Code Agent]
        B -->|planning| E[Planning Agent]
        B -->|general| F[Memory Agent]
        C --> F[Memory Agent]
        D --> F
        E --> F
        F --> G[Final Response]
    ```

    ### Tool Usage
    - **Theory Agent**: Uses `search_wikipedia` and `get_wikipedia_article` tools for theoretical concepts
    - **Code Agent**: Uses `execute_code`, `search_arxiv`, and `download_arxiv_paper` tools for code and research
    - **Planning Agent**: Uses `get_current_time`, `create_study_plan`, `list_study_materials`, and `read_study_material` tools
    - **Memory Agent**: Uses `save_memory` and `load_memory` tools to persist session data

    ### Memory Management
    - Session history stored in state and persisted to JSON file
    - Previous interactions loaded and used to improve routing and responses
    - Memory influences agent responses through context injection
    """)
    return


@app.cell
def _():
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    import arxiv
    import wikipedia
    from langchain.tools import tool
    from logly import logger

    return Path, arxiv, datetime, json, logger, os, tool, wikipedia


@app.cell
def _(Path, arxiv, datetime, json, logger, tool, wikipedia):
    MEMORY_FILE = Path("lab2_memory.json")
    STUDY_MATERIALS_DIR = Path("study_materials")
    STUDY_MATERIALS_DIR.mkdir(parents=True, exist_ok=True)

    @tool
    def save_memory(session_history: str, user_preferences: str) -> str:
        """Save session history and user preferences to persistent storage"""
        try:
            data = {
                "session_history": json.loads(session_history),
                "user_preferences": json.loads(user_preferences),
                "last_updated": datetime.now().isoformat(),
            }
            MEMORY_FILE.write_text(json.dumps(data, indent=2))
            logger.debug(f"Memory saved: {len(data['session_history'])} sessions")
            return f"Memory saved successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return f"Error saving memory: {str(e)}"

    @tool
    def load_memory() -> str:
        """Load session history and user preferences from persistent storage"""
        try:
            if not MEMORY_FILE.exists():
                logger.debug("No existing memory file found")
                return json.dumps({"session_history": [], "user_preferences": {}})

            data = json.loads(MEMORY_FILE.read_text())
            logger.debug(
                f"Memory loaded: {len(data.get('session_history', []))} sessions"
            )
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return json.dumps({"session_history": [], "user_preferences": {}})

    @tool
    def execute_code(code: str) -> str:
        """Execute Python code and return result"""
        logger.debug(f"Executing code: {code[:50]}...")
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            result = local_vars.get("result", "Code executed successfully")
            logger.debug(f"Code execution result: {str(result)[:100]}")
            return str(result)
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return f"Error: {str(e)}"

    @tool
    def get_current_time() -> str:
        """Get current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tool
    def create_study_plan(topic: str, duration_hours: int) -> str:
        """Create structured study plan for given topic and duration"""
        logger.debug(f"Creating study plan for: {topic}, {duration_hours}h")
        plan = {
            "topic": topic,
            "duration_hours": duration_hours,
            "schedule": [],
            "resources": [],
        }

        hours_per_session = 2
        num_sessions = max(1, duration_hours // hours_per_session)

        for i in range(num_sessions):
            plan["schedule"].append(
                {
                    "session": i + 1,
                    "focus": f"Part {i + 1} of {topic}",
                    "duration": f"{hours_per_session} hours",
                    "start_time": f"Day {(i // 3) + 1}, Session {(i % 3) + 1}",
                }
            )

        plan["resources"].extend(
            [
                f"Online tutorial for {topic}",
                f"Academic papers on {topic}",
                f"Practical exercises for {topic}",
                "Interactive coding challenges",
            ]
        )

        return json.dumps(plan, indent=2)

    @tool
    def search_wikipedia(query: str, max_results: int = 3) -> str:
        """Search Wikipedia and return summaries of top articles for theoretical concepts"""
        logger.debug(f"Searching Wikipedia for: {query}")
        try:
            results = wikipedia.search(query, results=max_results)
            summaries = []
            for title in results[:max_results]:
                try:
                    summary = wikipedia.summary(title, sentences=2)
                    summaries.append(f"**{title}**: {summary}")
                except (
                    wikipedia.exceptions.DisambiguationError,
                    wikipedia.exceptions.PageError,
                ):
                    continue
            return (
                "\n\n".join(summaries)
                if summaries
                else f"No results found for '{query}'"
            )
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return f"Error searching Wikipedia: {str(e)}"

    @tool
    def get_wikipedia_article(title: str) -> str:
        """Get detailed content from a specific Wikipedia article"""
        logger.debug(f"Fetching Wikipedia article: {title}")
        try:
            page = wikipedia.page(title)
            file_path = STUDY_MATERIALS_DIR / f"{title.replace(' ', '_')}.txt"
            file_path.write_text(page.content, encoding="utf-8")
            summary = wikipedia.summary(title, sentences=5)
            return f"**{page.title}**\n\n{summary}\n\n(Full article saved to {file_path.name})"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Ambiguous title '{title}'. Options: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"Article '{title}' not found"
        except Exception as e:
            logger.error(f"Wikipedia article error: {e}")
            return f"Error: {str(e)}"

    @tool
    def search_arxiv(query: str, max_results: int = 3) -> str:
        """Search arXiv for research papers and return titles and abstracts"""
        logger.debug(f"Searching arXiv for: {query}")
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = []
            for paper in search.results():
                arxiv_id = paper.entry_id.split("/")[-1]
                results.append(
                    f"**{paper.title}** (ID: {arxiv_id})\n"
                    f"Authors: {', '.join([a.name for a in paper.authors[:3]])}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                    f"Abstract: {paper.summary[:300]}...\n"
                )
            return "\n\n".join(results) if results else f"No papers found for '{query}'"
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return f"Error searching arXiv: {str(e)}"

    @tool
    def download_arxiv_paper(arxiv_id: str) -> str:
        """Download a paper from arXiv by its ID"""
        logger.debug(f"Downloading arXiv paper: {arxiv_id}")
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results(), None)
            if paper:
                file_path = STUDY_MATERIALS_DIR / f"{arxiv_id}.pdf"
                paper.download_pdf(
                    dirpath=str(STUDY_MATERIALS_DIR), filename=file_path.name
                )
                return f"Downloaded '{paper.title}' to {file_path.name}"
            else:
                return f"Paper {arxiv_id} not found"
        except Exception as e:
            logger.error(f"arXiv download error: {e}")
            return f"Error downloading paper: {str(e)}"

    @tool
    def list_study_materials(folder: str = "study_materials") -> str:
        """List all downloaded study materials (articles, papers)"""
        logger.debug("Listing study materials")
        try:
            path = Path(folder)
            if not path.exists():
                return "No study materials folder found"
            files = [f.name for f in path.glob("*")]
            if not files:
                return "No study materials downloaded yet"
            return f"Study materials ({len(files)} files):\n" + "\n".join(
                [f"- {f}" for f in sorted(files)]
            )
        except Exception as e:
            logger.error(f"List materials error: {e}")
            return f"Error listing materials: {str(e)}"

    @tool
    def read_study_material(filename: str) -> str:
        """Read content from a downloaded study material file"""
        logger.debug(f"Reading study material: {filename}")
        try:
            file_path = STUDY_MATERIALS_DIR / filename
            if not file_path.exists():
                return f"File '{filename}' not found in study materials"
            if file_path.suffix == ".pdf":
                return f"PDF file '{filename}' downloaded. Use a PDF reader to view it."
            content = file_path.read_text(encoding="utf-8")
            return content[:2000] + ("..." if len(content) > 2000 else "")
        except Exception as e:
            logger.error(f"Read material error: {e}")
            return f"Error reading file: {str(e)}"

    return (
        create_study_plan,
        download_arxiv_paper,
        execute_code,
        get_current_time,
        get_wikipedia_article,
        list_study_materials,
        load_memory,
        read_study_material,
        save_memory,
        search_arxiv,
        search_wikipedia,
    )


@app.cell
def _():
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    return BaseModel, ChatPromptTemplate, Field, PydanticOutputParser


@app.cell
def _(BaseModel, Field):
    class QueryClassification(BaseModel):
        query_type: str = Field(
            ..., description="Type of query: 'theory', 'code', 'planning', or 'general'"
        )
        confidence: float = Field(
            ..., description="Confidence in classification (0.0-1.0)"
        )
        reasoning: str = Field(..., description="Reasoning behind the classification")

    class TheoryResponse(BaseModel):
        answer: str = Field(..., description="Detailed theoretical answer")
        references: list[str] = Field(
            default_factory=list, description="Supporting references"
        )
        key_concepts: list[str] = Field(
            default_factory=list, description="Key concepts covered"
        )

    class CodeResponse(BaseModel):
        solution: str = Field(..., description="Code solution or explanation")
        best_practices: list[str] = Field(
            default_factory=list, description="Relevant best practices"
        )
        potential_issues: list[str] = Field(
            default_factory=list, description="Potential issues to watch for"
        )

    class PlanningResponse(BaseModel):
        plan: dict = Field(..., description="Structured plan with tasks and timeline")
        recommendations: list[str] = Field(
            default_factory=list, description="Additional recommendations"
        )
        estimated_duration: str = Field(..., description="Estimated time required")

    class MemoryUpdate(BaseModel):
        session_summary: str = Field(..., description="Summary of current session")
        user_preferences_updated: dict = Field(
            default_factory=dict, description="Updated user preferences"
        )
        action_items: list[str] = Field(
            default_factory=list, description="Follow-up action items"
        )

    class MultiAgentState(BaseModel):
        query: str = Field(..., description="User's input query")
        classification: QueryClassification | None = None
        theory_response: TheoryResponse | None = None
        code_response: CodeResponse | None = None
        planning_response: PlanningResponse | None = None
        memory_update: MemoryUpdate | None = None
        session_history: list[dict] = Field(
            default_factory=list, description="Session history"
        )
        user_preferences: dict = Field(
            default_factory=dict, description="User preferences"
        )
        final_response: str | None = None
        errors: list[str] = Field(
            default_factory=list, description="Any errors encountered"
        )
        active_agent: str | None = Field(None, description="Currently active agent")

    return (
        CodeResponse,
        MemoryUpdate,
        MultiAgentState,
        PlanningResponse,
        QueryClassification,
        TheoryResponse,
    )


@app.cell
def _():
    from dotenv import find_dotenv, load_dotenv
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    return (
        ChatOpenAI,
        END,
        HumanMessage,
        StateGraph,
        create_agent,
        find_dotenv,
        load_dotenv,
    )


@app.cell
def _(find_dotenv, load_dotenv, os):
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
        temperature=0.7,
    )
    return (llm,)


@app.cell
def _(ChatPromptTemplate, PydanticOutputParser, QueryClassification):
    router_parser = PydanticOutputParser(pydantic_object=QueryClassification)

    ROUTER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a query router for a multi-agent study assistant system.

            Analyze the user's query and classify it into one of these categories:
            - 'theory': Conceptual/theoretical questions about MAS, LLMs, AI, machine learning, or academic topics
            - 'code': Programming, implementation, debugging, or technical coding questions
            - 'planning': Task planning, study schedules, time management, or productivity questions
            - 'general': Questions that don't fit the above categories

            Consider previous session context if provided to improve classification accuracy.

            Provide your classification with confidence score (0.0-1.0) and clear reasoning.

            {format_instructions}
            /no_think
            """.strip(),
            ),
            ("human", "Previous context: {context}\n\nQuery: {query}"),
        ]
    )
    return ROUTER_PROMPT, router_parser


@app.cell
def _(ChatPromptTemplate, PydanticOutputParser, TheoryResponse):
    theory_parser = PydanticOutputParser(pydantic_object=TheoryResponse)

    THEORY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a theory expert specializing in Multi-Agent Systems, LLMs, and AI/ML concepts.

            Provide detailed, academic answers to theoretical questions. Include:
            - Clear explanations of concepts with proper terminology
            - Relevant references or sources (papers, books, researchers)
            - Key concepts and their relationships
            - Practical implications where relevant
            - Concrete examples to illustrate abstract concepts

            Consider the user's previous questions to provide continuity in explanations.

            {format_instructions}
            /no_think
            """.strip(),
            ),
            ("human", "Previous context: {context}\n\nQuestion: {query}"),
        ]
    )
    return THEORY_PROMPT, theory_parser


@app.cell
def _(ChatPromptTemplate, CodeResponse, PydanticOutputParser):
    code_parser = PydanticOutputParser(pydantic_object=CodeResponse)

    CODE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a code expert specializing in Python, LangChain, and LangGraph.

            Provide practical code solutions with:
            - Working, executable code examples
            - Best practices and design patterns
            - Potential issues, edge cases, and how to handle them
            - Clear explanations of the code logic
            - Type hints and documentation where appropriate

            You have access to tools to execute code. Use them when helpful to demonstrate solutions.

            Consider previous coding context to maintain consistency.

            {format_instructions}
            /no_think
            """.strip(),
            ),
            ("human", "Previous context: {context}\n\nQuestion: {query}"),
        ]
    )
    return CODE_PROMPT, code_parser


@app.cell
def _(ChatPromptTemplate, PlanningResponse, PydanticOutputParser):
    planning_parser = PydanticOutputParser(pydantic_object=PlanningResponse)

    PLANNING_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a planning expert specializing in study schedules and productivity.

            Create structured plans with:
            - Clear tasks broken into manageable steps
            - Realistic time estimates based on task complexity
            - Resource recommendations (books, courses, tools)
            - Actionable steps with priorities
            - Milestones and checkpoints

            You have access to tools for time management and plan creation. Use them appropriately.

            Consider previous planning requests to build on existing plans.

            {format_instructions}
            /no_think
            """.strip(),
            ),
            ("human", "Previous context: {context}\n\nRequest: {query}"),
        ]
    )
    return PLANNING_PROMPT, planning_parser


@app.cell
def _(ChatPromptTemplate, MemoryUpdate, PydanticOutputParser):
    memory_parser = PydanticOutputParser(pydantic_object=MemoryUpdate)

    MEMORY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a memory manager for the multi-agent system.

            Analyze the session and create:
            - Concise session summary capturing key points
            - Updated user preferences based on interaction patterns
            - Action items for follow-up or future reference

            Extract patterns like:
            - Topics of interest
            - Preferred response style (detailed/concise)
            - Learning goals
            - Technical skill level

            {format_instructions}
            /no_think
            """.strip(),
            ),
            (
                "human",
                "Session context: {context}\n\nPrevious memory: {previous_memory}",
            ),
        ]
    )
    return MEMORY_PROMPT, memory_parser


@app.cell
def _(
    create_agent,
    create_study_plan,
    download_arxiv_paper,
    execute_code,
    get_current_time,
    get_wikipedia_article,
    list_study_materials,
    llm,
    load_memory,
    read_study_material,
    save_memory,
    search_arxiv,
    search_wikipedia,
):
    theory_agent = create_agent(
        llm,
        [search_wikipedia, get_wikipedia_article],
        system_prompt="You are a theory expert. Use Wikipedia search tools to find and retrieve theoretical concepts, definitions, and explanations.",
    )

    code_agent = create_agent(
        llm,
        [execute_code, search_arxiv, download_arxiv_paper],
        system_prompt="You are a code expert. Use execute_code to demonstrate solutions and search_arxiv to find relevant research papers and implementations.",
    )

    planning_agent = create_agent(
        llm,
        [
            get_current_time,
            create_study_plan,
            list_study_materials,
            read_study_material,
        ],
        system_prompt="You are a planning expert. Use available tools for time management, plan creation, and tracking study materials.",
    )

    memory_agent = create_agent(
        llm,
        [save_memory, load_memory],
        system_prompt="You are a memory manager. Use save_memory and load_memory tools to manage session data.",
    )
    return code_agent, memory_agent, planning_agent, theory_agent


@app.cell
def _(ROUTER_PROMPT, datetime, llm, logger, router_parser):
    async def router_node(state):
        logger.debug(f"Router: Processing query: {state.query[:60]}...")

        context = ""
        if state.session_history:
            recent = state.session_history[-3:]
            context = "\n".join(
                [
                    f"- {h.get('classification', 'unknown')}: {h.get('query', '')[:50]}..."
                    for h in recent
                ]
            )

        messages = ROUTER_PROMPT.format_messages(
            format_instructions=router_parser.get_format_instructions(),
            context=context or "No previous context",
            query=state.query,
        )

        response = await llm.ainvoke(messages)
        classification = router_parser.parse(response.content)

        logger.debug(
            f"Router: Classified as '{classification.query_type}' (confidence: {classification.confidence:.2f})"
        )
        logger.debug(f"Router: Reasoning: {classification.reasoning}")

        return {
            "classification": classification,
            "session_history": state.session_history
            + [
                {
                    "query": state.query,
                    "classification": classification.query_type,
                    "confidence": classification.confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "active_agent": "router",
        }

    return (router_node,)


@app.cell
def _(HumanMessage, THEORY_PROMPT, llm, logger, theory_agent, theory_parser):
    async def theory_node(state):
        logger.debug(f"Theory Agent: ACTIVATED for query: {state.query[:60]}...")

        context = ""
        if state.session_history:
            theory_history = [
                h for h in state.session_history if h.get("classification") == "theory"
            ]
            if theory_history:
                context = "Previous theory topics: " + ", ".join(
                    [h.get("query", "")[:30] + "..." for h in theory_history[-2:]]
                )

        logger.debug(
            "Theory Agent: Using Wikipedia search tools for theoretical content"
        )
        tool_result = await theory_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Help answer this theoretical question: {state.query}. Use search_wikipedia to find relevant concepts and get_wikipedia_article for detailed information."
                    )
                ]
            }
        )
        logger.debug(
            f"Theory Agent: Tool execution result: {str(tool_result.get('messages', []))[:150]}"
        )

        messages = THEORY_PROMPT.format_messages(
            format_instructions=theory_parser.get_format_instructions(),
            context=context or "No previous theory context",
            query=state.query
            + f"\n\nWikipedia search results: {str(tool_result.get('messages', []))}",
        )

        response = await llm.ainvoke(messages)
        theory_response = theory_parser.parse(response.content)

        logger.debug(
            f"Theory Agent: Generated response with {len(theory_response.key_concepts)} key concepts"
        )
        logger.debug(
            f"Theory Agent: Key concepts: {', '.join(theory_response.key_concepts[:3])}"
        )

        return {"theory_response": theory_response, "active_agent": "theory"}

    return (theory_node,)


@app.cell
def _(CODE_PROMPT, HumanMessage, code_agent, code_parser, llm, logger):
    async def code_node(state):
        logger.debug(f"Code Agent: ACTIVATED for query: {state.query[:60]}...")

        context = ""
        if state.session_history:
            code_history = [
                h for h in state.session_history if h.get("classification") == "code"
            ]
            if code_history:
                context = "Previous code topics: " + ", ".join(
                    [h.get("query", "")[:30] + "..." for h in code_history[-2:]]
                )

        messages = CODE_PROMPT.format_messages(
            format_instructions=code_parser.get_format_instructions(),
            context=context or "No previous code context",
            query=state.query,
        )

        response = await llm.ainvoke(messages)
        code_response = code_parser.parse(response.content)

        logger.debug(
            f"Code Agent: Generated solution with {len(code_response.best_practices)} best practices"
        )

        if (
            "```python" in code_response.solution
            or "result =" in code_response.solution
        ):
            logger.debug("Code Agent: Attempting to execute code example via tool")
            try:
                code_to_exec = code_response.solution
                if "```python" in code_to_exec:
                    code_to_exec = (
                        code_to_exec.split("```python")[1].split("```")[0].strip()
                    )

                exec_result = await code_agent.ainvoke(
                    {
                        "messages": [
                            HumanMessage(content=f"Execute this code: {code_to_exec}")
                        ]
                    }
                )
                logger.debug(
                    f"Code Agent: Execution result: {str(exec_result.get('messages', []))[:100]}"
                )
            except Exception as e:
                logger.warning(f"Code Agent: Could not execute code: {e}")

        return {"code_response": code_response, "active_agent": "code"}

    return (code_node,)


@app.cell
def _(
    HumanMessage,
    PLANNING_PROMPT,
    llm,
    logger,
    planning_agent,
    planning_parser,
):
    async def planning_node(state):
        logger.debug(f"Planning Agent: ACTIVATED for query: {state.query[:60]}...")

        context = ""
        if state.session_history:
            planning_history = [
                h
                for h in state.session_history
                if h.get("classification") == "planning"
            ]
            if planning_history:
                context = "Previous plans: " + ", ".join(
                    [h.get("query", "")[:30] + "..." for h in planning_history[-2:]]
                )

        logger.debug("Planning Agent: Using time, planning, and study materials tools")
        tool_result = await planning_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Help with this planning request: {state.query}. Use get_current_time, create_study_plan, list_study_materials, and read_study_material tools as needed."
                    )
                ]
            }
        )
        logger.debug(
            f"Planning Agent: Tool execution result: {str(tool_result.get('messages', []))[:100]}"
        )

        messages = PLANNING_PROMPT.format_messages(
            format_instructions=planning_parser.get_format_instructions(),
            context=context or "No previous planning context",
            query=state.query
            + f"\n\nTool output: {str(tool_result.get('messages', []))}",
        )

        response = await llm.ainvoke(messages)
        planning_response = planning_parser.parse(response.content)

        logger.debug(
            f"Planning Agent: Created plan with duration: {planning_response.estimated_duration}"
        )
        logger.debug(
            f"Planning Agent: {len(planning_response.recommendations)} recommendations"
        )

        return {"planning_response": planning_response, "active_agent": "planning"}

    return (planning_node,)


@app.cell
def _(
    HumanMessage,
    MEMORY_PROMPT,
    json,
    llm,
    logger,
    memory_agent,
    memory_parser,
):
    async def memory_node(state):
        logger.debug(
            "Memory Agent: ACTIVATED - Processing session and generating final response"
        )

        logger.debug("Memory Agent: Loading existing memory via tool")
        memory_load_result = await memory_agent.ainvoke(
            {"messages": [HumanMessage(content="Load current memory")]}
        )
        previous_memory = str(memory_load_result.get("messages", [{}]))
        logger.debug(f"Memory Agent: Loaded memory: {previous_memory[:100]}")

        context_parts = [f"User query: {state.query}"]

        if state.classification:
            context_parts.append(
                f"Classification: {state.classification.query_type} (confidence: {state.classification.confidence})"
            )
            context_parts.append(f"Reasoning: {state.classification.reasoning}")

        if state.theory_response:
            context_parts.append(
                f"Theory response: {state.theory_response.answer[:150]}..."
            )
            context_parts.append(
                f"Key concepts: {', '.join(state.theory_response.key_concepts[:3])}"
            )

        if state.code_response:
            context_parts.append(
                f"Code response: {state.code_response.solution[:150]}..."
            )
            context_parts.append(
                f"Best practices: {', '.join(state.code_response.best_practices[:2])}"
            )

        if state.planning_response:
            context_parts.append(
                f"Planning response: {json.dumps(state.planning_response.plan)[:150]}..."
            )
            context_parts.append(
                f"Duration: {state.planning_response.estimated_duration}"
            )

        context = "\n".join(context_parts)

        messages = MEMORY_PROMPT.format_messages(
            format_instructions=memory_parser.get_format_instructions(),
            context=context,
            previous_memory=previous_memory,
        )

        response = await llm.ainvoke(messages)
        memory_update = memory_parser.parse(response.content)

        logger.debug(
            f"Memory Agent: Session summary: {memory_update.session_summary[:100]}"
        )
        logger.debug(f"Memory Agent: {len(memory_update.action_items)} action items")

        updated_history = state.session_history + [
            {"summary": memory_update.session_summary}
        ]
        updated_prefs = {
            **state.user_preferences,
            **memory_update.user_preferences_updated,
        }

        logger.debug("Memory Agent: Saving updated memory via tool")
        save_result = await memory_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Save this memory - history: {json.dumps(updated_history)}, preferences: {json.dumps(updated_prefs)}"
                    )
                ]
            }
        )
        logger.debug(
            f"Memory Agent: Save result: {str(save_result.get('messages', []))[:50]}"
        )

        final_response = "# Multi-Agent Assistant Response\n\n"

        if state.theory_response:
            final_response += (
                f"## Theoretical Analysis\n\n{state.theory_response.answer}\n\n"
            )
            if state.theory_response.key_concepts:
                final_response += f"**Key Concepts:** {', '.join(state.theory_response.key_concepts)}\n\n"
            if state.theory_response.references:
                final_response += (
                    "**References:**\n"
                    + "\n".join(
                        [f"- {ref}" for ref in state.theory_response.references]
                    )
                    + "\n\n"
                )

        elif state.code_response:
            final_response += f"## Code Solution\n\n{state.code_response.solution}\n\n"
            if state.code_response.best_practices:
                final_response += (
                    "**Best Practices:**\n"
                    + "\n".join(
                        [f"- {bp}" for bp in state.code_response.best_practices]
                    )
                    + "\n\n"
                )
            if state.code_response.potential_issues:
                final_response += (
                    "**Potential Issues:**\n"
                    + "\n".join(
                        [f"- {pi}" for pi in state.code_response.potential_issues]
                    )
                    + "\n\n"
                )

        elif state.planning_response:
            final_response += f"## Study Plan\n\n```json\n{json.dumps(state.planning_response.plan, indent=2)}\n```\n\n"
            final_response += f"**Estimated Duration:** {state.planning_response.estimated_duration}\n\n"
            if state.planning_response.recommendations:
                final_response += (
                    "**Recommendations:**\n"
                    + "\n".join(
                        [f"- {rec}" for rec in state.planning_response.recommendations]
                    )
                    + "\n\n"
                )

        else:
            final_response += (
                "I've analyzed your query. Please rephrase for better routing.\n\n"
            )

        final_response += (
            f"---\n\n**Session Summary:** {memory_update.session_summary}\n\n"
        )
        if memory_update.action_items:
            final_response += "**Action Items:**\n" + "\n".join(
                [f"- {item}" for item in memory_update.action_items]
            )

        logger.debug(
            f"Memory Agent: Final response length: {len(final_response)} chars"
        )

        return {
            "memory_update": memory_update,
            "final_response": final_response,
            "user_preferences": updated_prefs,
            "active_agent": "memory",
        }

    return (memory_node,)


@app.cell
def _(logger):
    def route_query(state) -> str:
        if not state.classification:
            logger.warning("Router: No classification found, routing to memory")
            return "memory"

        route = state.classification.query_type

        if route == "theory":
            logger.debug("Router: Routing to THEORY agent")
            return "theory"
        elif route == "code":
            logger.debug("Router: Routing to CODE agent")
            return "code"
        elif route == "planning":
            logger.debug("Router: Routing to PLANNING agent")
            return "planning"
        else:
            logger.debug("Router: Routing to MEMORY agent (general query)")
            return "memory"

    return (route_query,)


@app.cell
def _(MultiAgentState, StateGraph):
    workflow = StateGraph(MultiAgentState)
    return (workflow,)


@app.cell
def _(
    END,
    code_node,
    memory_node,
    planning_node,
    route_query,
    router_node,
    theory_node,
    workflow,
):
    workflow.add_node("router", router_node)
    workflow.add_node("theory", theory_node)
    workflow.add_node("code", code_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("memory", memory_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "theory": "theory",
            "code": "code",
            "planning": "planning",
            "memory": "memory",
        },
    )

    workflow.add_edge("theory", "memory")
    workflow.add_edge("code", "memory")
    workflow.add_edge("planning", "memory")

    workflow.add_edge("memory", END)

    app = workflow.compile()
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
    ## System Evaluation - Running Real Experiments

    ### Test Queries
    We'll test the system with 5 diverse queries covering all agent types.
    """)
    return


@app.cell
def _():
    test_queries = [
        {
            "id": 1,
            "query": "What are the key challenges in implementing multi-agent systems with LLMs?",
            "expected_agent": "theory",
            "description": "Conceptual question about MAS and LLMs",
        },
        {
            "id": 2,
            "query": "How can I implement a router pattern in LangGraph with conditional edges?",
            "expected_agent": "code",
            "description": "Programming question about LangGraph implementation",
        },
        {
            "id": 3,
            "query": "Create a 10-hour study plan for learning about multi-agent systems",
            "expected_agent": "planning",
            "description": "Planning request for study schedule",
        },
        {
            "id": 4,
            "query": "What are the differences between supervisor and sequential workflow patterns in MAS?",
            "expected_agent": "theory",
            "description": "Theoretical comparison question",
        },
        {
            "id": 5,
            "query": "Write Python code that uses LangChain tools with proper error handling",
            "expected_agent": "code",
            "description": "Code implementation request with specific requirements",
        },
    ]
    return (test_queries,)


@app.cell
def _(MultiAgentState, app, logger):
    async def run_test_query(query_text: str):
        logger.debug(f"Starting test query: {query_text[:60]}...")
        state = MultiAgentState(query=query_text)
        result = await app.ainvoke(state)
        logger.debug(f"Test completed. Active agent: {result.get('active_agent')}")
        return result

    return (run_test_query,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Running Experiments

    Execute the cell below to run all 5 test queries and collect results.
    """)
    return


@app.cell
async def _(logger, run_test_query, test_queries):
    logger.debug("Starting experiment batch")

    experiment_results = []

    for test in test_queries:
        logger.debug(f"Test #{test['id']}: {test['description']}")
        result = await run_test_query(test["query"])

        experiment_results.append(
            {
                "test_id": test["id"],
                "query": test["query"],
                "description": test["description"],
                "expected_agent": test["expected_agent"],
                "actual_agent": result.get("classification").query_type
                if result.get("classification")
                else "unknown",
                "confidence": result.get("classification").confidence
                if result.get("classification")
                else 0.0,
                "reasoning": result.get("classification").reasoning
                if result.get("classification")
                else "",
                "response_preview": result.get("final_response")[:200] + "..."
                if result.get("final_response")
                else "",
                "memory_summary": result.get("memory_update").session_summary
                if result.get("memory_update")
                else "",
                "tools_used": result.get("active_agent"),
            }
        )

    logger.debug("Experiment batch completed")

    experiment_results
    return


if __name__ == "__main__":
    app.run()
