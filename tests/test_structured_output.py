from agents.structured_output import StructuredOutput


tool_calls = [
    {
        "tool": "rag_search",
        "arguments": {
            "query": "What is RAG?"
        },
        "result": {
            "success": True
        }
    },
    {
        "tool": "calculator",
        "arguments": {
            "expression": "250 * 18 / 100"
        },
        "result": {
            "success": True,
            "result": 45
        }
    }
]


result = StructuredOutput.build(
    answer=(
        "RAG combines retrieval with "
        "language model generation."
    ),
    tool_calls=tool_calls,
    grounded=True,
    confidence=0.94,
    citations=[
        {
            "source": "rag_notes.txt",
            "page": 1
        }
    ]
)


print("\n==============================")
print("STRUCTURED AGENT OUTPUT")
print("==============================")

print(
    StructuredOutput.to_json(result)
)