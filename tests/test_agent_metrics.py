from evaluation.agent_metrics import AgentMetrics


def test_agent_metrics():

    results = [
        {
            "success": True,
            "grounded": True,
            "confidence": 0.90,
            "evidence": [1, 2],
            "tools_used": ["rag_search"],
        },
        {
            "success": True,
            "grounded": True,
            "confidence": 0.80,
            "evidence": [1],
            "tools_used": ["rag_search"],
        },
        {
            "success": False,
            "grounded": False,
            "confidence": 0.20,
            "evidence": [],
            "tools_used": [],
        },
    ]

    metrics = AgentMetrics().evaluate(results)

    assert metrics["total"] == 3
    assert metrics["successful"] == 2
    assert metrics["success_rate"] == 0.6667
    assert metrics["grounded"] == 2
    assert metrics["grounded_rate"] == 0.6667
    assert metrics["average_confidence"] == 0.6333
    assert metrics["average_evidence"] == 1.0
    assert metrics["tool_usage_rate"] == 0.6667

    print("\n==============================")
    print("AGENT EVALUATION METRICS")
    print("==============================")

    for key, value in metrics.items():
        print(f"{key}: {value}")