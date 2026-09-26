from agents.orchestrator import AgentOrchestrator
from rag.pipeline import RAGPipeline
from rag.ingestion import DocumentIngestion


print("\n==============================")
print("ORCHESTRATOR TEST")
print("==============================")


# -------------------------------------------------
# 1. Create RAG pipeline
# -------------------------------------------------

retriever = RAGPipeline()


# -------------------------------------------------
# 2. Load knowledge document
# -------------------------------------------------

with open(
    "data/rag_notes.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# -------------------------------------------------
# 3. Convert document into chunks
# -------------------------------------------------

ingestion = DocumentIngestion(
    chunk_size=500,
    overlap=100
)

chunks = ingestion.ingest(
    text=text,
    source="rag_notes.txt",
    page=1
)

print(
    f"\nINDEXED CHUNKS: {len(chunks)}"
)


# -------------------------------------------------
# 4. Build RAG index
# -------------------------------------------------

retriever.build(chunks)


print("\nRAG INDEX: READY")


# -------------------------------------------------
# 5. Create Agent Orchestrator
# -------------------------------------------------

orchestrator = AgentOrchestrator(
    retriever
)


# -------------------------------------------------
# 6. User request
# -------------------------------------------------

request = {
    "query": "What is RAG?"
}


print("\nQUERY:")
print(request["query"])


# -------------------------------------------------
# 7. Run agent
# -------------------------------------------------

result = orchestrator.run(
    request
)


# -------------------------------------------------
# 8. Display result
# -------------------------------------------------

print("\n==============================")
print("ORCHESTRATOR RESULT")
print("==============================")

print(result)


# -------------------------------------------------
# 9. Structured output
# -------------------------------------------------

if isinstance(result, dict):

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(
        result.get(
            "answer",
            "No answer returned."
        )
    )

    print("\n==============================")
    print("GROUNDED")
    print("==============================")

    print(
        result.get(
            "grounded",
            False
        )
    )

    print("\n==============================")
    print("CONFIDENCE")
    print("==============================")

    print(
        result.get(
            "confidence",
            0.0
        )
    )

    print("\n==============================")
    print("CITATIONS")
    print("==============================")

    print(
        result.get(
            "citations",
            []
        )
    )

    print("\n==============================")
    print("TOOLS USED")
    print("==============================")

    print(
        result.get(
            "tools_used",
            []
        )
    )