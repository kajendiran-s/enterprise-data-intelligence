
from src.agent.agent_loop import EnterpriseAgent


agent = EnterpriseAgent()

question = "How does Spark distribute work across executors?"

result = agent.run(question)

print("=" * 80)
print("QUESTION")
print("=" * 80)
print(result["question"])

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print(result["answer"])

print("\n" + "=" * 80)
print("TOOL TRACE")
print("=" * 80)

for step in result["tool_trace"]:
    print(f"\nIteration: {step['iteration']}")
    print(f"Tool: {step['tool']}")
    print(f"Arguments: {step['arguments']}")
    print(f"Result: {step['result'][:500]}")

print(f"\nStatus: {result['status']}")
print(f"Iterations: {result['iterations']}")
