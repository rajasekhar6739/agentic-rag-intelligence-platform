from rag.pipeline import RAGPipeline
from agents.schemas import DocumentChunk


documents = [

    DocumentChunk(
        text=(
            "API latency increased because database "
            "connections were exhausted. Connection pool "
            "limits were reached during the incident."
        ),
        source="incident.pdf",
        page=10,
        chunk_id=0
    ),

    DocumentChunk(
        text=(
            "The deployment process requires engineering "
            "approval before production release."
        ),
        source="runbook.pdf",
        page=5,
        chunk_id=0
    ),

    DocumentChunk(
        text=(
            "CPU utilization increased significantly "
            "during the incident, reaching 92 percent."
        ),
        source="metrics.pdf",
        page=3,
        chunk_id=0
    )
]


pipeline = RAGPipeline()

pipeline.build(documents)


query = (
    "Why did API latency increase "
    "during the incident?"
)


result = pipeline.search(
    query,
    candidate_k=3,
    top_k=2
)


print("\n==============================")
print("ORIGINAL QUERY")
print("==============================")
print(result["original_query"])


print("\n==============================")
print("REWRITTEN QUERY")
print("==============================")
print(result["rewritten_query"])


print("\n==============================")
print("KEYWORDS")
print("==============================")

for keyword in result["keywords"]:
    print("-", keyword)


print("\n==============================")
print("EVIDENCE")
print("==============================")


for item in result["evidence"]:

    print("\nSOURCE:", item["source"])
    print("PAGE:", item["page"])
    print(
        "RERANK SCORE:",
        round(
            item["rerank_score"],
            4
        )
    )
    print("TEXT:", item["text"])