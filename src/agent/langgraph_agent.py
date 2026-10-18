
import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.tools import AgentTools
from src.agent.sql_tool import run_readonly_sql
from src.agent.citation_validator import validate_agent_answer


SYSTEM_PROMPT = """
You are an enterprise data intelligence assistant.

TOOL SELECTION:
1. Use search_knowledge_base for technical information in enterprise documents.
2. Use query_sales_data for sales, revenue, products, and regions.
3. Use get_system_capabilities for questions about project capabilities.
4. Use tools when evidence is needed. Do not invent tool results.

CITATION RULES:
1. For knowledge-base answers, use only exact [Source N] IDs
   supplied in the knowledge-base tool's source_map.
2. For SQL answers, use the exact evidence ID returned by the SQL tool,
   formatted as (Evidence: SQL-001), replacing it with the actual ID.
3. Never invent citation or evidence IDs.
4. Do not use RAG citations to support SQL-only answers.
5. If evidence is insufficient, state the limitation.

Answer clearly and concisely.
"""

# State is the shared memory of this graph execution.
# MessagesState already manages the conversation messages.
# We extend it to retain the validation result as well.
class AgentState(MessagesState):
    validation: dict[str, Any]


def build_agent():
    agent_tools = AgentTools()

    @tool
    def search_knowledge_base(question: str) -> str:
        """Search enterprise documents for technical information."""
        return agent_tools.search_knowledge_base(question)

    @tool
    def get_system_capabilities() -> str:
        """Retrieve structured metadata about this project's capabilities."""
        return agent_tools.get_system_capabilities()

    @tool
    def query_sales_data(sql_query: str) -> str:
        """Execute a read-only SQL query against the sales table.

        The sales table contains sale_id, product, region,
        quantity, and unit_price. Revenue is quantity * unit_price.
        Use this tool for questions about sales, revenue, products,
        or regions.
        """
        result = run_readonly_sql(sql_query)
        return json.dumps(result, ensure_ascii=False, default=str)

    tools = [
        search_knowledge_base,
        get_system_capabilities,
        query_sales_data,
    ]

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
    ).bind_tools(tools)

    # NODE 1: Ask the model to answer or request a tool.
    def agent_node(state: AgentState):
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ])
        return {"messages": [response]}

    # NODE 2: Validate the final answer against the latest
    # SQL or knowledge-base tool result.
    def validation_node(state: AgentState):
        final_answer = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage)
                and not message.tool_calls
                and isinstance(message.content, str)
                and message.content.strip()
            ),
            "",
        )

        latest_evidence = None
        evidence_type = None

        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                continue

            if message.name not in {
                "query_sales_data",
                "search_knowledge_base",
                "get_system_capabilities",
            }:
                continue

            try:
                latest_evidence = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                # Capability metadata may be plain text rather than JSON.
                latest_evidence = (
                    {"raw": message.content}
                    if message.name == "get_system_capabilities"
                    and isinstance(message.content, str)
                    else None
                )

            if message.name == "query_sales_data":
                evidence_type = "sql"
            elif message.name == "search_knowledge_base":
                evidence_type = "rag"
            else:
                evidence_type = "capabilities"

            break

        if not isinstance(latest_evidence, dict):
            return {
                "validation": {
                    "valid": False,
                    "error": "No usable SQL or knowledge-base evidence found.",
                }
            }

        if evidence_type == "sql":
            validation = validate_agent_answer(
                answer=final_answer,
                evidence_type="sql",
                sql_tool_result=latest_evidence,
            )
        elif evidence_type == "rag":
            validation = validate_agent_answer(
                answer=final_answer,
                evidence_type="rag",
                source_map=latest_evidence.get("source_map", []),
            )
        elif evidence_type == "capabilities":
            validation = {
                "valid": (
                    bool(final_answer.strip())
                    and bool(latest_evidence)
                ),
                "evidence_type": "capabilities",
                "checks": {
                    "tool_result_present": bool(latest_evidence),
                    "non_empty_answer": bool(final_answer.strip()),
                },
            }
        return {"validation": validation}

    # NODE 3: Avoid presenting an answer as verified if its
    # evidence checks fail.
    def validation_failed_node(state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't verify my answer against the available "
                        "evidence. I won't present it as confirmed. "
                        "Please refine the question or check the underlying "
                        "data and source documents."
                    )
                )
            ]
        }

    # ROUTING: A valid answer ends the graph.
    # An invalid answer goes to the safe failure node.
    def route_after_validation(state: AgentState):
        if state.get("validation", {}).get("valid") is True:
            return "end"

        return "validation_failed"

    # Build the workflow graph.
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("validate", validation_node)
    builder.add_node("validation_failed", validation_failed_node)

    builder.add_edge(START, "agent")

    # The model either requests a tool or produces a final answer.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: "validate",
        },
    )

    # After a tool executes, return its result to the model.
    builder.add_edge("tools", "agent")

    # Validate before completing the graph.
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "end": END,
            "validation_failed": "validation_failed",
        },
    )

    builder.add_edge("validation_failed", END)

    return builder.compile()
