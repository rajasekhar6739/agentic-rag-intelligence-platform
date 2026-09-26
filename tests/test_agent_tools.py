from agents.function_caller import FunctionCaller


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
                        "RAG combines information retrieval "
                        "with language model generation."
                    ),
                    "source": "rag_notes.txt",
                    "page": 1,
                    "chunk_id": 0
                },
                "rerank_score": 4.5
            },
            {
                "document": {
                    "text": (
                        "Hybrid retrieval can combine "
                        "BM25 and vector search."
                    ),
                    "source": "retrieval.txt",
                    "page": 2,
                    "chunk_id": 1
                },
                "rerank_score": 4.1
            }
        ]


retriever = FakeRetriever()

agent = FunctionCaller(
    retriever
)


result = agent.run(
    "What is RAG and calculate 250 * 18 percent."
)


print("\n==============================")
print("AGENTIC TOOL TEST")
print("==============================")


print("\nANSWER:")
print(result.get("answer", ""))


print("\nSTRUCTURED OUTPUT:")
print(result)


print("\nTOOLS USED:")

for tool in result.get(
    "tools_used",
    []
):
    print("-", tool)


print("\nCITATIONS:")

for citation in result.get(
    "citations",
    []
):
    print("-", citation)