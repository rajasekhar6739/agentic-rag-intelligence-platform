from agents.tool_agent import ToolAgent
from agents.schemas import DocumentChunk


def test_agent_rag():
    documents = [
        DocumentChunk(
            text=(
                "API latency increased because database "
                "connections were exhausted. The connection "
                "pool reached its configured limit."
            ),
            source="incident.pdf",
            page=10,
            chunk_id=0,
        ),
        DocumentChunk(
            text=(
                "CPU utilization reached 92 percent during "
                "the production incident."
            ),
            source="metrics.pdf",
            page=3,
            chunk_id=1,
        ),
        DocumentChunk(
            text=(
                "Production deployments require engineering "
                "approval before release."
            ),
            source="runbook.pdf",
            page=5,
            chunk_id=2,
        ),
    ]

    agent = ToolAgent()

    agent.load_documents(documents)

    request = """
    Using our internal documents, explain why API latency
    increased during the incident. Include the relevant
    evidence and source.
    """

    result = agent.run(request)

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(result["answer"])

    print("\n==============================")
    print("ITERATIONS")
    print("==============================")
    print(result["iterations"])

    print("\n==============================")
    print("EXECUTION TRACE")
    print("==============================")

    for event in result["execution_trace"]:
        print("\nIteration:", event["iteration"])
        print("Tool:", event["tool"])
        print("Arguments:", event["arguments"])
        print("Result:")
        print(event["result"])

    assert result is not None
    assert "answer" in result
    assert result["answer"]