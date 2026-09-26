from agents.orchestrator import AgentOrchestrator
from agents.schemas import DocumentChunk


documents = [

    DocumentChunk(
        text=(
            "Database connections were exhausted during "
            "the production incident. The exhausted "
            "connection pool caused API requests to wait "
            "for available database connections, resulting "
            "in increased API latency."
        ),
        source="incident.pdf",
        page=10,
        chunk_id=0
    ),

    DocumentChunk(
        text=(
            "CPU utilization reached 92 percent during "
            "the production incident."
        ),
        source="metrics.pdf",
        page=3,
        chunk_id=1
    )
]


orchestrator = AgentOrchestrator()

orchestrator.load_documents(
    documents
)


request = """
Using the internal incident documents, explain
why API latency increased and provide the source.
"""


result = orchestrator.run(
    request,
    max_retries=2
)


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(result["answer"])


print("\n==============================")
print("ACCEPTED")
print("==============================")

print(result["accepted"])


print("\n==============================")
print("ATTEMPTS")
print("==============================")


for attempt in result["attempts"]:

    print(
        "\nAttempt:",
        attempt["attempt"]
    )

    print(
        "Answer:",
        attempt["answer"]
    )

    print(
        "Critique:",
        attempt["critique"]
    )