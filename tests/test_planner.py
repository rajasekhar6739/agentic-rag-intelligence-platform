from agents.planner import AgentPlanner


planner = AgentPlanner()


result = planner.plan(
    "Explain RAG from my documents "
    "and calculate 250 * 18 percent."
)


print("\n==============================")
print("AGENT PLAN")
print("==============================")


print(result)


print("\nTASKS:")

for task in result.get(
    "tasks",
    []
):
    print(
        task["id"],
        "->",
        task["tool"],
        "->",
        task["description"]
    )