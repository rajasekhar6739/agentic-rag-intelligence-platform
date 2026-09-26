from rag.reranker import Reranker


results = [
    {
        "document": {
            "source": "doc1",
            "page": 1,
            "chunk_id": 0,
            "text": (
                "Transformers use self-attention "
                "to model relationships between tokens."
            )
        },
        "rrf_score": 0.03
    },
    {
        "document": {
            "source": "doc2",
            "page": 1,
            "chunk_id": 0,
            "text": (
                "BM25 is a lexical information "
                "retrieval algorithm."
            )
        },
        "rrf_score": 0.02
    }
]


reranker = Reranker()


ranked = reranker.rerank(
    query="How does self attention work in transformers?",
    results=results,
    top_k=2
)


print("\n==============================")
print("RERANKER")
print("==============================")


for item in ranked:

    print(
        "Rerank score:",
        item["rerank_score"]
    )

    print(
        "Document:",
        item["document"]
    )

    print()