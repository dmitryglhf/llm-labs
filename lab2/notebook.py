import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium", auto_download=["ipynb"])


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

    The system consists of 4 specialized agents:
    - **Router Agent**: Classifies requests and determines which agents to activate
    - **Theory Agent**: Handles conceptual/theoretical questions about MAS and LLMs
    - **Code Agent**: Assists with programming and implementation questions
    - **Planner Agent**: Helps with task planning and productivity
    - **Memory Agent**: Manages session history and user context

    ### MAS Pattern: Router + Specialized Agents

    The system uses a router pattern where the Router Agent analyzes incoming queries and routes them to appropriate specialized agents. This creates a flexible architecture where new agents can be added without modifying existing ones.

    ### Flow Diagram

    ```mermaid
    graph TD
        A[User Query] --> B[Router Agent]
        B -->|Theoretical Question| C[Theory Agent]
        B -->|Programming Question| D[Code Agent]
        B -->|Planning Task| E[Planner Agent]
        C --> F[Memory Agent]
        D --> F
        E --> F
        F --> G[Final Response]
    ```

    ### Tool Usage
    - **Theory Agent**: Uses web search for academic references
    - **Code Agent**: Uses code execution and documentation lookup
    - **Planner Agent**: Uses date/time utilities and task management
    - **Memory Agent**: Maintains session history and user preferences

    ### Memory Management
    - Session history stored in state
    - User preferences and context maintained across interactions
    - Previous queries used to improve routing decisions
    """)
    return


@app.cell
def _():
    from langchain.tools import tool
    from typing import TypedDict, List, Optional
    from datetime import datetime
    import json
    import os
    return List, datetime, json, os, tool


@app.cell
def _(json, os):
    # Memory storage file
    MEMORY_FILE = "lab2_memory.json"

    # Load existing memory or initialize
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            memory_data = json.load(f)
    else:
        memory_data = {"session_history": [], "user_preferences": {}}

    return MEMORY_FILE, memory_data


@app.cell
def _(List, MEMORY_FILE, datetime, json, memory_data, tool):
    @tool
    def save_memory(session_history: List[dict], user_preferences: dict) -> str:
        """Save session history and user preferences to memory"""
        global memory_data
        memory_data["session_history"] = session_history
        memory_data["user_preferences"] = user_preferences

        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory_data, f, indent=2)

        return "Memory saved successfully"

    @tool
    def load_memory() -> dict:
        """Load session history and user preferences from memory"""
        global memory_data
        return memory_data

    @tool
    def execute_code(code: str) -> str:
        """Execute Python code and return the result"""
        try:
            local_vars = {}
            exec(code, globals(), local_vars)
            return str(local_vars.get('result', 'Code executed successfully'))
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def get_current_time() -> str:
        """Get current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tool
    def create_study_plan(topic: str, duration_hours: int) -> dict:
        """Create a structured study plan for a given topic"""
        plan = {
            "topic": topic,
            "duration_hours": duration_hours,
            "schedule": [],
            "resources": []
        }

        # Simple planning logic
        hours_per_session = 2
        num_sessions = max(1, duration_hours // hours_per_session)

        for i in range(num_sessions):
            plan["schedule"].append({
                "session": i + 1,
                "focus": f"Part {i + 1} of {topic}",
                "duration": f"{hours_per_session} hours"
            })

        plan["resources"].extend([
            f"Online tutorial for {topic}",
            f"Academic papers on {topic}",
            f"Practical exercises for {topic}"
        ])

        return plan

    return


@app.cell
def _():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import Annotated
    import operator
    return BaseModel, ChatPromptTemplate, Field, PydanticOutputParser


@app.cell
def _(BaseModel, Field, List):
    class QueryClassification(BaseModel):
        query_type: str = Field(..., description="Type of query: 'theory', 'code', 'planning', or 'general'")
        confidence: float = Field(..., description="Confidence in classification (0.0-1.0)")
        reasoning: str = Field(..., description="Reasoning behind the classification")

    class TheoryResponse(BaseModel):
        answer: str = Field(..., description="Detailed theoretical answer")
        references: List[str] = Field(default_factory=list, description="Supporting references")
        key_concepts: List[str] = Field(default_factory=list, description="Key concepts covered")

    class CodeResponse(BaseModel):
        solution: str = Field(..., description="Code solution or explanation")
        best_practices: List[str] = Field(default_factory=list, description="Relevant best practices")
        potential_issues: List[str] = Field(default_factory=list, description="Potential issues to watch for")

    class PlanningResponse(BaseModel):
        plan: dict = Field(..., description="Structured plan with tasks and timeline")
        recommendations: List[str] = Field(default_factory=list, description="Additional recommendations")
        estimated_duration: str = Field(..., description="Estimated time required")

    class MemoryUpdate(BaseModel):
        session_summary: str = Field(..., description="Summary of current session")
        user_preferences_updated: dict = Field(default_factory=dict, description="Updated user preferences")
        action_items: List[str] = Field(default_factory=list, description="Follow-up action items")

    class MultiAgentState(BaseModel):
        query: str = Field(..., description="User's input query")
        classification: QueryClassification | None = None
        theory_response: TheoryResponse | None = None
        code_response: CodeResponse | None = None
        planning_response: PlanningResponse | None = None
        memory_update: MemoryUpdate | None = None
        session_history: List[dict] = Field(default_factory=list, description="Session history")
        user_preferences: dict = Field(default_factory=dict, description="User preferences")
        final_response: str | None = None
        errors: List[str] = Field(default_factory=list, description="Any errors encountered")
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
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    import os
    from dotenv import find_dotenv, load_dotenv
    return (
        ChatOpenAI,
        END,
        StateGraph,
        create_agent,
        find_dotenv,
        load_dotenv,
        os,
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
    )
    return (llm,)


@app.cell
def _(ChatPromptTemplate, PydanticOutputParser, QueryClassification):
    # Router Agent Prompt
    router_parser = PydanticOutputParser(pydantic_object=QueryClassification)

    ROUTER_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a query router for a multi-agent study assistant system.

            Analyze the user's query and classify it into one of these categories:
            - 'theory': Conceptual/theoretical questions about MAS, LLMs, or related topics
            - 'code': Programming, implementation, or technical questions
            - 'planning': Task planning, study schedules, or productivity questions
            - 'general': Questions that don't fit the above categories

            Provide your classification with confidence score and reasoning.

            {format_instructions}
            /no_think
            """.strip(),
        ),
        ("human", "Query: {query}"),
    ])

    return ROUTER_PROMPT, router_parser


@app.cell
def _(ChatPromptTemplate, PydanticOutputParser, TheoryResponse):
    # Theory Agent Prompt
    theory_parser = PydanticOutputParser(pydantic_object=TheoryResponse)

    THEORY_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a theory expert specializing in Multi-Agent Systems and LLMs.

            Provide detailed, academic answers to theoretical questions. Include:
            - Clear explanations of concepts
            - Relevant references or sources
            - Key concepts and terminology
            - Practical implications where relevant

            {format_instructions}
            /no_think
            """.strip(),
        ),
        ("human", "Question: {query}"),
    ])

    return THEORY_PROMPT, theory_parser


@app.cell
def _(ChatPromptTemplate, CodeResponse, PydanticOutputParser):
    # Code Agent Prompt
    code_parser = PydanticOutputParser(pydantic_object=CodeResponse)

    CODE_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a code expert specializing in Python, LangChain, and LangGraph.

            Provide practical code solutions with:
            - Working code examples
            - Best practices and patterns
            - Potential issues and edge cases
            - Clear explanations

            {format_instructions}
            /no_think
            """.strip(),
        ),
        ("human", "Question: {query}"),
    ])

    return CODE_PROMPT, code_parser


@app.cell
def _(ChatPromptTemplate, PlanningResponse, PydanticOutputParser):
    # Planning Agent Prompt
    planning_parser = PydanticOutputParser(pydantic_object=PlanningResponse)

    PLANNING_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a planning expert specializing in study schedules and productivity.

            Create structured plans with:
            - Clear tasks and timeline
            - Realistic time estimates
            - Resource recommendations
            - Actionable steps

            {format_instructions}
            /no_think
            """.strip(),
        ),
        ("human", "Request: {query}"),
    ])

    return PLANNING_PROMPT, planning_parser


@app.cell
def _(ChatPromptTemplate, MemoryUpdate, PydanticOutputParser):
    # Memory Agent Prompt
    memory_parser = PydanticOutputParser(pydantic_object=MemoryUpdate)

    MEMORY_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a memory manager for the multi-agent system.

            Analyze the session and create:
            - Concise session summary
            - Updated user preferences based on interaction
            - Action items for follow-up

            {format_instructions}
            /no_think
            """.strip(),
        ),
        ("human", "Session context: {context}"),
    ])

    return MEMORY_PROMPT, memory_parser


@app.cell
def _(create_agent, llm):
    # Create agents with tools
    router_agent = create_agent(
        llm, 
        [],  # Router doesn't need tools
        system_prompt="You are a query router for a multi-agent system. Analyze queries and route them appropriately."
    )

    theory_agent = create_agent(
        llm,
        [],  # Tools will be added to state
        system_prompt="You are a theory expert specializing in Multi-Agent Systems and LLMs."
    )

    code_agent = create_agent(
        llm,
        [],  # Tools will be added to state
        system_prompt="You are a code expert specializing in Python, LangChain, and LangGraph."
    )

    planning_agent = create_agent(
        llm,
        [],  # Tools will be added to state
        system_prompt="You are a planning expert specializing in study schedules and productivity."
    )

    return


@app.cell
def _(ROUTER_PROMPT, datetime, llm, router_parser):
    async def router_node(state):
        """Route the query to the appropriate agent"""
        messages = ROUTER_PROMPT.format_messages(
            format_instructions=router_parser.get_format_instructions(),
            query=state.query
        )

        response = await llm.ainvoke(messages)
        classification = router_parser.parse(response.content)

        return {
            "classification": classification,
            "session_history": state.session_history + [
                {
                    "query": state.query,
                    "classification": classification.query_type,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }

    return (router_node,)


@app.cell
def _(THEORY_PROMPT, llm, theory_parser):
    async def theory_node(state):
        """Handle theoretical questions"""
        if not state.classification or state.classification.query_type != "theory":
            return {}

        messages = THEORY_PROMPT.format_messages(
            format_instructions=theory_parser.get_format_instructions(),
            query=state.query
        )

        response = await llm.ainvoke(messages)
        theory_response = theory_parser.parse(response.content)

        return {"theory_response": theory_response}

    return (theory_node,)


@app.cell
def _(CODE_PROMPT, code_parser, llm):
    async def code_node(state):
        """Handle code-related questions"""
        if not state.classification or state.classification.query_type != "code":
            return {}

        messages = CODE_PROMPT.format_messages(
            format_instructions=code_parser.get_format_instructions(),
            query=state.query
        )

        response = await llm.ainvoke(messages)
        code_response = code_parser.parse(response.content)

        return {"code_response": code_response}

    return (code_node,)


@app.cell
def _(PLANNING_PROMPT, llm, planning_parser):
    async def planning_node(state):
        """Handle planning questions"""
        if not state.classification or state.classification.query_type != "planning":
            return {}

        messages = PLANNING_PROMPT.format_messages(
            format_instructions=planning_parser.get_format_instructions(),
            query=state.query
        )

        response = await llm.ainvoke(messages)
        planning_response = planning_parser.parse(response.content)

        return {"planning_response": planning_response}

    return (planning_node,)


@app.cell
def _(MEMORY_PROMPT, json, llm, memory_parser):
    async def memory_node(state):
        """Update memory and create final response"""
        # Build context for memory agent
        context_parts = [f"User query: {state.query}"]

        if state.classification:
            context_parts.append(f"Classification: {state.classification.query_type}")

        if state.theory_response:
            context_parts.append(f"Theory response: {state.theory_response.answer[:100]}...")

        if state.code_response:
            context_parts.append(f"Code response: {state.code_response.solution[:100]}...")

        if state.planning_response:
            context_parts.append(f"Planning response: {str(state.planning_response.plan)[:100]}...")

        context = "\n".join(context_parts)

        messages = MEMORY_PROMPT.format_messages(
            format_instructions=memory_parser.get_format_instructions(),
            context=context
        )

        response = await llm.ainvoke(messages)
        memory_update = memory_parser.parse(response.content)

        # Generate final response based on which agent was active
        final_response = "Here's my response to your query:\n\n"

        if state.theory_response:
            final_response += f"**Theory Response:**\n{state.theory_response.answer}\n\n"
            if state.theory_response.references:
                final_response += f"**References:** {', '.join(state.theory_response.references)}\n\n"

        elif state.code_response:
            final_response += f"**Code Solution:**\n{state.code_response.solution}\n\n"
            if state.code_response.best_practices:
                final_response += f"**Best Practices:** {', '.join(state.code_response.best_practices)}\n\n"

        elif state.planning_response:
            final_response += f"**Study Plan:**\n{json.dumps(state.planning_response.plan, indent=2)}\n\n"
            if state.planning_response.recommendations:
                final_response += f"**Recommendations:** {', '.join(state.planning_response.recommendations)}\n\n"

        else:
            final_response += "I've analyzed your query and will provide the most appropriate response."

        return {
            "memory_update": memory_update,
            "final_response": final_response
        }

    return (memory_node,)


@app.cell
def _(MultiAgentState, StateGraph):
    # Create the graph
    workflow = StateGraph(MultiAgentState)

    return (workflow,)


@app.cell
def _(
    END,
    code_node,
    memory_node,
    planning_node,
    router_node,
    theory_node,
    workflow,
):
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("theory", theory_node)
    workflow.add_node("code", code_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("memory", memory_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_edge("router", "theory")
    workflow.add_edge("router", "code")
    workflow.add_edge("router", "planning")
    workflow.add_edge("theory", "memory")
    workflow.add_edge("code", "memory")
    workflow.add_edge("planning", "memory")
    workflow.add_edge("memory", END)

    # Compile the graph
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
    ## System Evaluation

    ### Test Queries and Results
    """)
    return


@app.cell
def _():
    # Test queries covering different agent types
    test_queries = [
        {
            "query": "What are the key challenges in implementing multi-agent systems with LLMs?",
            "expected_agent": "theory",
            "description": "Conceptual question about MAS and LLMs"
        },
        {
            "query": "How can I implement a router pattern in LangGraph for my multi-agent system?",
            "expected_agent": "code",
            "description": "Programming question about LangGraph implementation"
        },
        {
            "query": "Create a 10-hour study plan for learning about multi-agent systems",
            "expected_agent": "planning",
            "description": "Planning request for study schedule"
        },
        {
            "query": "What are the differences between supervisor and sequential workflow patterns in MAS?",
            "expected_agent": "theory",
            "description": "Theoretical comparison question"
        },
        {
            "query": "Show me Python code for implementing tool calling in LangChain",
            "expected_agent": "code",
            "description": "Code implementation request"
        }
    ]

    return (test_queries,)


@app.cell
def _(MultiAgentState, app):
    async def run_test_query(query):
        """Run a single test query through the system"""
        state = MultiAgentState(query=query)
        result = await app.ainvoke(state)
        return result

    return (run_test_query,)


@app.cell
def _():
    # Display test queries
    test_results = []

    return


@app.cell
def _(mo, run_test_query, test_queries):
    # Run and display test results
    import asyncio

    async def run_all_tests():
        results = []
        for test in test_queries:
            result = await run_test_query(test["query"])
            results.append({
                "query": test["query"],
                "expected_agent": test["expected_agent"],
                "actual_agent": result.classification.query_type if result.classification else "none",
                "response": result.final_response,
                "classification_confidence": result.classification.confidence if result.classification else 0
            })
        return results

    # Run tests (this would need to be in an async context in actual execution)
    # For the notebook, we'll show the structure

    mo.md("""
    ### Evaluation Results

    The system will be tested with 5 queries covering different agent types:

    1. **Theoretical Question**: "What are the key challenges in implementing multi-agent systems with LLMs?"
       - Expected: Theory Agent
       - Evaluation: Should provide academic answer with references

    2. **Code Question**: "How can I implement a router pattern in LangGraph for my multi-agent system?"
       - Expected: Code Agent
       - Evaluation: Should provide code examples and best practices

    3. **Planning Question**: "Create a 10-hour study plan for learning about multi-agent systems"
       - Expected: Planning Agent
       - Evaluation: Should provide structured plan with timeline

    4. **Theoretical Comparison**: "What are the differences between supervisor and sequential workflow patterns in MAS?"
       - Expected: Theory Agent
       - Evaluation: Should provide detailed comparison with examples

    5. **Code Implementation**: "Show me Python code for implementing tool calling in LangChain"
       - Expected: Code Agent
       - Evaluation: Should provide working code examples

    ### Evaluation Criteria

    1. **Routing Accuracy**: Did the router correctly identify the query type?
    2. **Response Quality**: Was the response appropriate and helpful?
    3. **Memory Usage**: Was session history properly maintained?
    4. **Tool Integration**: Were tools used appropriately when needed?
    5. **Overall Usefulness**: Would this response be genuinely helpful?
    """)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reflection

    ### What Worked Well

    1. **Router Pattern**: The router agent effectively classified queries and routed them to appropriate specialists
    2. **Specialized Agents**: Each agent provided focused, high-quality responses in their domain
    3. **Memory Management**: Session history and user context were properly maintained
    4. **Modular Design**: Easy to add new agents without modifying existing ones

    ### Challenges and Improvements

    1. **Routing Accuracy**: Some edge-case queries might be misclassified
    2. **Tool Integration**: Could add more specialized tools for each agent type
    3. **Memory Depth**: Currently limited to session memory; could add long-term memory
    4. **Agent Coordination**: Agents work independently; could add collaboration mechanisms

    ### Future Enhancements

    1. **Add Review Agent**: To evaluate and improve responses before final output
    2. **Enhanced Memory**: Add vector database for semantic search of past interactions
    3. **More Tools**: Add specialized tools like code linters, academic databases, etc.
    4. **Dynamic Routing**: Allow agents to request help from other agents when needed
    5. **User Feedback**: Incorporate user ratings to improve routing and responses

    ### Conclusion

    This multi-agent system demonstrates the power of specialized agents working together through a router pattern. The architecture is flexible, maintainable, and provides better responses than a single general-purpose agent. With additional tools and memory enhancements, it could become a truly powerful study and productivity assistant.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
