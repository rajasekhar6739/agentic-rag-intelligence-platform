from agents.task_graph import TaskGraph


print()
print("=" * 60)
print("AURA TASK GRAPH TEST")
print("=" * 60)


# ============================================================
# CREATE GRAPH
# ============================================================

graph = TaskGraph()


# ============================================================
# TASK 1
# ============================================================

graph.add_task(
    task_id=1,
    tool="rag_search",
    description="Retrieve RAG evidence.",
    arguments={
        "query": "What is RAG?"
    }
)


# ============================================================
# TASK 2
# ============================================================

graph.add_task(
    task_id=2,
    tool="analyzer",
    description="Analyze retrieved evidence.",
    arguments={
        "mode": "summarize"
    },
    depends_on=[1]
)


# ============================================================
# FAKE EXECUTOR
# ============================================================

def fake_executor(
    tool_name,
    arguments,
):

    print()
    print("EXECUTING:", tool_name)
    print("ARGUMENTS:", arguments)

    if tool_name == "rag_search":

        return {
            "success": True,
            "tool": "rag_search",
            "evidence": [
                "RAG combines retrieval with generation."
            ],
        }

    if tool_name == "analyzer":

        dependency_results = arguments.get(
            "_dependency_results",
            {},
        )

        return {
            "success": True,
            "tool": "analyzer",
            "analysis": (
                "The retrieved evidence "
                "supports the RAG definition."
            ),
            "dependency_results": dependency_results,
        }

    return {
        "success": False,
        "tool": tool_name,
        "error": "Unknown tool",
    }


# ============================================================
# CHECK TASKS
# ============================================================

print()
print("REGISTERED TASKS")
print("=" * 60)

for task in graph.list_tasks():
    print(task)


# ============================================================
# CHECK EXECUTION ORDER
# ============================================================

print()
print("EXECUTION ORDER")
print("=" * 60)

order = graph.execution_order()

print(order)

assert order == [1, 2]


# ============================================================
# EXECUTE GRAPH
# ============================================================

result = graph.execute(
    executor=fake_executor
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("TASK GRAPH RESULT")
print("=" * 60)

print(
    "SUCCESS:",
    result["success"],
)

print(
    "EXECUTION ORDER:",
    result["execution_order"],
)

print(
    "RESULTS:",
    result["results"],
)


# ============================================================
# ASSERTIONS
# ============================================================

assert result["success"] is True

assert result["execution_order"] == [
    1,
    2,
]

assert result["results"][1]["success"] is True

assert result["results"][2]["success"] is True


# ============================================================
# VERIFY DEPENDENCY PASSING
# ============================================================

task_2_result = result["results"][2]

assert (
    "dependency_results"
    in task_2_result
)

assert (
    1
    in task_2_result["dependency_results"]
)

assert (
    task_2_result["dependency_results"][1]["success"]
    is True
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("TASK GRAPH TEST PASSED")
print("=" * 60)
print()