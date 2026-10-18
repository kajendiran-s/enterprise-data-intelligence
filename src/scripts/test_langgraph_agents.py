import json
from src.agent.citation_validator import validate_final_citations, validate_sql_evidence, validate_sql_answer_consistency, validate_agent_answer
from langchain_core.messages import AIMessage, ToolMessage
from src.agent.langgraph_agent import build_agent

agent = build_agent()
# question = "Which region has the highest total sales revenue?"
question = "How does Spark distribute work across executors?"
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": question}
        ]
    },
    config={
        "recursion_limit": 12,
    },
)

print("\nGRAPH VALIDATION")
print(json.dumps(
    result.get("validation", {}),
    indent=2,
))

messages = result["messages"]

print("=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\n" + "=" * 80)
print("AGENT TRACE")
print("=" * 80)

for message in messages:
    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            print(
                f"Tool requested: {tool_call['name']}"
                f"\nArguments: {tool_call['args']}\n"
            )

    elif isinstance(message, ToolMessage):
        print(f"Tool executed: {message.name}")
        print("Full tool result:")
        print(message.content)
        print()

final_message = messages[-1]

print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print(final_message.content)

# # Collect source mappings from the executed knowledge-base tools.
# source_map = []
# sql_tool_result = None
# for message in messages:
#     if isinstance(message, ToolMessage) and message.name == "search_knowledge_base":
#         try:
#             tool_data = json.loads(message.content)
#             source_map.extend(tool_data.get("source_map", []))
#         except (json.JSONDecodeError, TypeError):
#             print("Warning: Could not parse tool output as JSON.")
#     if message.name == "query_sales_data":
#         sql_tool_result = json.loads(message.content)

# citation_result = validate_final_citations(
#     answer=str(final_message.content),
#     source_map=source_map,
# )

# print("\n" + "=" * 80)
# print("CITATION VALIDATION")
# print("=" * 80)
# print(json.dumps(citation_result, indent=2))

# if result.get('validation',{}).get("evidence_type","") == "sql":
#     validation = validate_sql_evidence(
#         final_message.content,
#         sql_tool_result,
#     )

#     print("\nSQL EVIDENCE VALIDATION")
#     print(validation)

#     consistency = validate_sql_answer_consistency(
#         final_message.content,
#         sql_tool_result,
#     )

#     print("\nSQL RESULT CONSISTENCY")
#     print(consistency)

# validation = validate_agent_answer(
#     answer=final_message.content,
#     evidence_type="sql",
#     sql_tool_result=sql_tool_result,
# )

# print("\nOVERALL ANSWER VALIDATION")
# print(validation)


