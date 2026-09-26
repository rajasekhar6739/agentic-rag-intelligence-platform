from agents.plan_executor import PlanExecutor


class FakeRetriever:

    def search(
        self,
        query,
        top_k=5
    ):

        return [
            {
                "document": {
                    "text": (
                        "RAG combines retrieval "
                        "with generation."
                    ),
                    "source": "rag.txt",
                    "page": 1,
                    "chunk_id": 0
                },
                "rerank_score": 4.5
            }
        ]


executor = PlanExecutor(
    FakeRetriever()
)


plan = {
    "tasks": [
        {
            "id": 1,
            "description": "What is RAG?",
            "tool": "rag_search"
        },
        {
            "id": 2,
            "description": "250 * 18 / 100",
            "tool": "calculator"
        }
    ]
}


results = executor.execute(
    plan
)


print("\n==============================")
print("PLAN EXECUTION")
print("==============================")


for item in results:

    print("\nTask:", item["task_id"])
    print("Tool:", item["tool"])
    print("Arguments:", item["arguments"])
    print("Result:", item["result"])