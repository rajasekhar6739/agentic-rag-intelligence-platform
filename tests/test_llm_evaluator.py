from evaluation.llm_evaluator import LLMEvaluator


def test_llm_evaluator():

    evaluator = LLMEvaluator()

    question = (
        "Why did API latency increase?"
    )

    evidence = [
        {
            "text": (
                "API latency increased because "
                "database connections were exhausted."
            ),
            "source": "incident.pdf",
            "page": 10,
        }
    ]

    answer = (
        "API latency increased because "
        "database connections were exhausted."
    )

    result = evaluator.evaluate(
        question=question,
        answer=answer,
        evidence=evidence,
        citations=["[incident.pdf, p. 10]"],
        tools_used=["rag_search"],
    )

    assert result["success"] is True
    assert "scores" in result
    assert "overall" in result["scores"]
    assert result["grounded"] is True