from agents.function_caller import FunctionCaller


class FakeRetriever:

    def search(self, query, top_k=5):

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


agent = FunctionCaller(
    FakeRetriever()
)


first = agent.run(
    "What is RAG?"
)

print("\nFIRST:")
print(first["answer"])


second = agent.run(
    "Explain that again in one sentence."
)

print("\nSECOND:")
print(second["answer"])


print("\nMEMORY:")
print(
    agent.memory.get_history()
)