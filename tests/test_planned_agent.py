from agents.planned_agent import PlannedAgent


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
                        "RAG combines information "
                        "retrieval with generation."
                    ),
                    "source": "rag.txt",
                    "page": 1,
                    "chunk_id": 0
                },
                "rerank_score": 4.5
            }
        ]


agent = PlannedAgent(
    FakeRetriever()
)


result = agent.run(
    "Explain RAG from my documents "
    "and calculate 250 * 18 percent."
)


print("\n==============================")
print("FULL AGENT WORKFLOW")
print("==============================")


print("\nANSWER:")
print(result["answer"])


print("\nGROUNDED:")
print(result["grounded"])


print("\nCONFIDENCE:")
print(result["confidence"])


print("\nTOOLS:")
print(result["tools_used"])


print("\nPLAN:")
print(result["plan"])


print("\nRESULTS:")

for item in result["results"]:

    print(
        "\nTask:",
        item["task_id"]
    )

    print(
        "Tool:",
        item["tool"]
    )

    print(
        "Result:",
        item["result"]
    )