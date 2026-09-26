from agents.orchestrator import AgentOrchestrator


print()
print("=" * 60)
print("AURA MULTI-TOOL TEST")
print("=" * 60)


orchestrator = AgentOrchestrator()


query = "Calculate 125 * 8 + 50 / 2"


print()
print("QUERY:")
print(query)


result = orchestrator.run(
    query
)


print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    "ANSWER:",
    result.get("answer")
)

print(
    "TOOLS:",
    result.get("tools_used")
)

print(
    "GROUNDED:",
    result.get("grounded")
)

print(
    "CONFIDENCE:",
    result.get("confidence")
)

assert "calculator" in result.get(
    "tools_used",
    []
)

print()
print("MULTI-TOOL TEST PASSED")