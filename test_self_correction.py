from agents.orchestrator import AgentOrchestrator
from rag.pipeline import RAGPipeline
from rag.ingestion import DocumentIngestion


print()
print("=" * 60)
print("AURA SELF-CORRECTION TEST")
print("=" * 60)


# =========================================================
# RAG
# =========================================================

retriever = RAGPipeline()

with open(
    "data/rag_notes.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


ingestion = DocumentIngestion(
    chunk_size=500,
    overlap=100
)

chunks = ingestion.ingest(
    text=text,
    source="rag_notes.txt",
    page=1
)

print()
print("INDEXED CHUNKS:", len(chunks))

retriever.build(chunks)

print()
print("RAG INDEX: READY")


# =========================================================
# ORCHESTRATOR
# =========================================================

orchestrator = AgentOrchestrator(
    retriever=retriever
)


# =========================================================
# QUERY
# =========================================================

query = (
    "What are the detailed production deployment "
    "steps, infrastructure requirements, monitoring "
    "architecture, scaling strategy, and security "
    "configuration for RAG?"
)

print()
print("QUERY:")
print(query)


# =========================================================
# RUN
# =========================================================

result = orchestrator.run(query)


# =========================================================
# RESULT
# =========================================================

print()
print("=" * 60)
print("SELF-CORRECTION RESULT")
print("=" * 60)

print()
print("ANSWER:")
print(result.get("answer"))

print()
print("GROUNDED:")
print(result.get("grounded"))

print()
print("CONFIDENCE:")
print(result.get("confidence"))

print()
print("FINAL ATTEMPT:")
print(result.get("final_attempt"))

print()
print("SELF CORRECTED:")
print(result.get("self_corrected"))


# =========================================================
# ATTEMPTS
# =========================================================

attempts = result.get("attempts", [])

print()
print("=" * 60)
print("ATTEMPT DETAILS")
print("=" * 60)

for attempt in attempts:

    print()
    print("ATTEMPT:", attempt.get("attempt"))
    print("GROUNDED:", attempt.get("grounded"))
    print("CONFIDENCE:", attempt.get("confidence"))

    evidence = attempt.get("evidence", [])

    print("EVIDENCE:", len(evidence))

    critic = attempt.get("critic", {})

    print(
        "CRITIC SUPPORTED:",
        critic.get("supported", False)
    )

    print(
        "CRITIC ISSUES:",
        critic.get("issues", [])
    )


# =========================================================
# VALIDATION
# =========================================================

assert isinstance(result, dict)

assert "attempts" in result

assert result.get("final_attempt", 0) >= 1


print()
print("=" * 60)
print("SELF-CORRECTION TEST PASSED")
print("=" * 60)