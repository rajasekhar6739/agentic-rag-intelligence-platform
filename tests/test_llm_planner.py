from agents.llm_planner import LLMPlanner
from llm.client import LLMClient


def test_llm_planner_fallback_rag():

    planner = LLMPlanner(
        llm=LLMClient(
            api_key=None
        )
    )

    result = planner.plan(
        """
        Using our internal documents,
        explain why API latency increased.
        """
    )

    assert result["success"] is True
    assert result["tool_calls"]

    assert (
        result["tool_calls"][0]["tool"]
        == "rag_search"
    )


def test_llm_planner_fallback_calculator():

    planner = LLMPlanner(
        llm=LLMClient(
            api_key=None
        )
    )

    result = planner.plan(
        "Calculate 1250 multiplied by 18 percent."
    )

    assert result["success"] is True
    assert result["tool_calls"]

    assert (
        result["tool_calls"][0]["tool"]
        == "calculator"
    )