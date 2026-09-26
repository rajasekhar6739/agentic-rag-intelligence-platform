from agents.tool_agent import ToolAgent


def evaluate(result):
    answer = result.get("answer", "")
    evidence = result.get("evidence", [])
    citations = result.get("citations", [])
    tools = result.get("tools_used", [])

    grounded = bool(result.get("grounded"))
    has_evidence = len(evidence) > 0
    has_citations = len(citations) > 0
    has_tools = len(tools) > 0

    scores = {
        "grounded": int(grounded),
        "evidence": int(has_evidence),
        "citations": int(has_citations),
        "tool_usage": int(has_tools),
        "answer_nonempty": int(bool(answer.strip())),
    }

    overall = sum(scores.values()) / len(scores)

    return {
        "scores": scores,
        "overall_score": round(overall, 3),
        "status": "PASS" if overall >= 0.8 else "NEEDS_REVIEW",
    }


def main():
    agent = ToolAgent()

    documents = [
        {
            "text": (
                "API latency increased because database "
                "connections were exhausted. The connection "
                "pool reached its configured limit."
            ),
            "source": "incident.pdf",
            "page": 10,
        }
    ]

    agent.load_documents(documents)

    result = agent.run(
        "Why did API latency increase during the incident?"
    )

    evaluation = evaluate(result)

    print("=" * 60)
    print("LLM / AGENT EVALUATION")
    print("=" * 60)

    print("\nAnswer:")
    print(result["answer"])

    print("\nEvaluation:")
    print(evaluation)

    print("\nOverall score:", evaluation["overall_score"])
    print("Status:", evaluation["status"])


if __name__ == "__main__":
    main()