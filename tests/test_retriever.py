from rag.retriever import HybridRetriever
from agents.schemas import DocumentChunk


documents = [

    DocumentChunk(
        text=(
            "API latency increased because "
            "database connections were exhausted."
        ),
        source="incident.pdf",
        page=10,
        chunk_id=0
    ),

    DocumentChunk(
        text=(
            "The deployment process requires "
            "engineering approval."
        ),
        source="runbook.pdf",
        page=5,
        chunk_id=0
    ),

    DocumentChunk(
        text=(
            "CPU utilization increased significantly "
            "during the incident."
        ),
        source="metrics.pdf",
        page=3,
        chunk_id=0
    )
]


retriever = HybridRetriever()

retriever.build(documents)


results = retriever.search(
    "database latency incident",
    top_k=3
)


for result in results:

    print("\n-------------------------")

    print(
        "SOURCE:",
        result["source"]
    )

    print(
        "PAGE:",
        result["page"]
    )

    print(
        "VECTOR SCORE:",
        round(
            result["vector_score"],
            4
        )
    )

    print(
        "BM25 SCORE:",
        round(
            result["bm25_score"],
            4
        )
    )

    print(
        "HYBRID SCORE:",
        round(
            result["hybrid_score"],
            4
        )
    )

    print(
        "TEXT:",
        result["text"]
    )