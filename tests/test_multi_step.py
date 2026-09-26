from agents.orchestrator import AgentOrchestrator


print()
print("=" * 60)
print("AURA MULTI-STEP AGENT TEST")
print("=" * 60)


class FakeRetriever:

    def search(self, query, top_k=5):

        return {
            "original_query": query,
            "evidence": [
                {
                    "text": (
                        "RAG combines information retrieval "
                        "with a large language model."
                    ),
                    "source": "rag_notes.txt",
                    "page": 1,
                    "chunk_id": 0,
                    "score": 0.95
                },
                {
                    "text": (
                        "A RAG pipeline can contain ingestion, "
                        "chunking, embeddings, vector search, "
                        "keyword retrieval, reranking, "
                        "and answer generation."
                    ),
                    "source": "rag_notes.txt",
                    "page": 1,
                    "chunk_id": 1,
                    "score": 0.91
                }
            ]
        }


retriever = FakeRetriever()


orchestrator = AgentOrchestrator(
    retriever=retriever
)


query = (
    "Explain the RAG architecture "
    "and analyze its main components."
)


print()
print("QUERY:")
print(query)


result = orchestrator.run(
    query,
    max_attempts=1
)


print()
print("=" * 60)
print("MULTI-STEP RESULT")
print("=" * 60)


print()
print("ANSWER:")
print(result.get("answer"))


print()
print("PLAN:")
print(result.get("plan"))


print()
print("TASK GRAPH:")
print(result.get("task_graph"))


print()
print("TOOLS:")
print(result.get("tools_used"))


print()
print("GROUNDED:")
print(result.get("grounded"))


# =========================================================
# ASSERTIONS
# =========================================================

assert result["success"] is True

plan = result["plan"]

tasks = plan.get(
    "tasks",
    []
)

assert len(tasks) >= 2

assert tasks[0]["tool"] == "rag_search"

assert tasks[1]["tool"] == (
    "evidence_analyzer"
)

assert tasks[1].get(
    "depends_on"
) == [1]


graph = result.get(
    "task_graph",
    {}
)

assert graph.get(
    "success"
) is True

execution_order = graph.get(
    "execution_order",
    []
)

assert execution_order == [1, 2]

graph_results = graph.get(
    "results",
    {}
)

assert 1 in graph_results
assert 2 in graph_results

assert graph_results[1]["success"] is True
assert graph_results[2]["success"] is True


analysis = graph_results[2]

assert analysis["evidence_count"] > 0


print()
print("=" * 60)
print("MULTI-STEP TEST PASSED")
print("=" * 60)
