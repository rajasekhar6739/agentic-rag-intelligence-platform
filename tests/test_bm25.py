from rag.bm25 import BM25Store
from agents.schemas import DocumentChunk


documents = [

    DocumentChunk(
        text="API latency increased because database connections were exhausted.",
        source="incident.pdf",
        page=10,
        chunk_id=0
    ),

    DocumentChunk(
        text="The deployment process requires engineering approval.",
        source="runbook.pdf",
        page=5,
        chunk_id=0
    ),

    DocumentChunk(
        text="CPU utilization increased significantly during the incident.",
        source="metrics.pdf",
        page=3,
        chunk_id=0
    )
]


store = BM25Store()

store.build(documents)

results = store.search(
    "database connections latency",
    top_k=2
)


for result in results:

    print("\nSOURCE:", result["source"])
    print("PAGE:", result["page"])
    print("SCORE:", result["score"])
    print("TEXT:", result["text"])