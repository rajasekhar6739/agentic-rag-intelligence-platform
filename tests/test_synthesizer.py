from agents.synthesizer import ResultSynthesizer


synthesizer = ResultSynthesizer()


results = [
    {
        "task_id": 1,
        "tool": "rag_search",
        "result": {
            "success": True,
            "results": [
                {
                    "text": (
                        "RAG combines retrieval "
                        "with generation."
                    ),
                    "source": "rag.txt",
                    "page": 1
                }
            ]
        }
    },
    {
        "task_id": 2,
        "tool": "calculator",
        "result": {
            "success": True,
            "expression": "250 * 18 / 100",
            "result": 45
        }
    }
]


result = synthesizer.synthesize(
    query=(
        "What is RAG and what is "
        "18 percent of 250?"
    ),
    results=results
)


print("\n==============================")
print("SYNTHESIZER")
print("==============================")

print(result)