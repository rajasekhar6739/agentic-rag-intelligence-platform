from rag.hybrid_retriever import HybridRetriever


class FakeStore:

    def __init__(self, results):
        self.results = results

    def search(
        self,
        query,
        top_k=5
    ):
        return self.results[:top_k]


bm25 = FakeStore([
    {
        "source": "doc1",
        "page": 1,
        "chunk_id": 0,
        "text": "BM25 result"
    },
    {
        "source": "doc2",
        "page": 1,
        "chunk_id": 0,
        "text": "Second result"
    }
])


vector = FakeStore([
    {
        "source": "doc2",
        "page": 1,
        "chunk_id": 0,
        "text": "Second result"
    },
    {
        "source": "doc3",
        "page": 2,
        "chunk_id": 0,
        "text": "Vector result"
    }
])


retriever = HybridRetriever(
    bm25_store=bm25,
    vector_store=vector
)


results = retriever.search(
    "transformer attention",
    top_k=3
)


print("\n==============================")
print("HYBRID RETRIEVAL")
print("==============================")


for result in results:

    print(
        result["rrf_score"],
        result["document"]
    )