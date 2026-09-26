from agents.orchestrator import AgentOrchestrator


print()
print("=" * 60)
print("AURA TASK GRAPH ORCHESTRATOR TEST")
print("=" * 60)


class FakeRetriever:

    def search(
        self,
        query,
        top_k=5
    ):

        return {
            "original_query": query,
            "evidence": [
                {
                    "text": (
                        "RAG combines retrieval "
                        "with language generation."
                    ),
                    "source": "test.txt",
                    "page": 1,
                    "chunk_id": 0,
                    "score": 0.9
                }
            ]
        }


retriever = FakeRetriever()

orchestrator = AgentOrchestrator(
    retriever=retriever
)


plan = {
    "query": "What is RAG?",
    "tasks": [
        {
            "id": 1,
            "description": "Retrieve evidence.",
            "tool": "rag_search",
            "arguments": {
                "query": "What is RAG?",
                "top_k": 5
            }
        }
    ]
}


result = orchestrator._execute_task_graph(
    plan=plan,
    query="What is RAG?"
)


print()
print("EXECUTION ORDER:")
print(
    result["execution_order"]
)

print()
print("RESULTS:")
print(
    result["results"]
)


assert result["success"] is True
assert result["execution_order"] == [1]

assert 1 in result["results"]

assert result["results"][1]["success"] is True


print()
print("TASK GRAPH ORCHESTRATOR TEST PASSED")