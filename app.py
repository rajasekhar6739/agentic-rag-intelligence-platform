from agents.tool_agent import ToolAgent


def main():
    print("=" * 60)
    print("AI AGENTIC RAG SYSTEM")
    print("=" * 60)

    agent = ToolAgent()

    print("\nAvailable tools:")
    print(agent.list_tools())

    request = input("\nEnter your question: ").strip()

    if not request:
        print("Question cannot be empty.")
        return

    result = agent.run(request)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.get("answer", ""))

    print("\n" + "=" * 60)
    print("GROUNDING")
    print("=" * 60)
    print("Grounded:", result.get("grounded"))
    print("Confidence:", result.get("confidence"))

    print("\n" + "=" * 60)
    print("TOOLS USED")
    print("=" * 60)
    for tool in result.get("tools_used", []):
        print("-", tool)

    print("\n" + "=" * 60)
    print("CITATIONS")
    print("=" * 60)
    for citation in result.get("citations", []):
        print("-", citation)

    print("\n" + "=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)

    for event in result.get("execution_trace", []):
        print(
            f"\nIteration {event.get('iteration')} "
            f"| Tool: {event.get('tool')}"
        )
        print("Arguments:", event.get("arguments"))
        print("Result:", event.get("result"))


if __name__ == "__main__":
    main()