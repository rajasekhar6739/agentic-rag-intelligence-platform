from evaluation.report import EvaluationReport


report = EvaluationReport()


retrieved = [
    "doc_1",
    "doc_3",
    "doc_2"
]

relevant = [
    "doc_1",
    "doc_2"
]

execution_trace = [
    {
        "result": {
            "success": True
        }
    },
    {
        "result": {
            "success": True
        }
    }
]

attempts = [
    {
        "latency_ms": 800
    },
    {
        "latency_ms": 600
    }
]

results = [
    {
        "accepted": True
    }
]


result = report.generate(
    retrieved=retrieved,
    relevant=relevant,
    execution_trace=execution_trace,
    attempts=attempts,
    results=results
)


print("\n==============================")
print("UNIFIED EVALUATION REPORT")
print("==============================")

print(result)