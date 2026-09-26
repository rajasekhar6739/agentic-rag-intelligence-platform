from agents.tool_agent import ToolAgent


def test_agent_planner_execution():

    agent = ToolAgent()

    plan = agent.plan(
        "Calculate 1250 multiplied by 18 percent."
    )

    assert plan["success"] is True
    assert plan["tool_calls"]

    result = agent.execute_plan(
        plan
    )

    assert result["success"] is True
    assert result["tool_calls"]

    first = result["tool_calls"][0]

    assert first["tool"] == "calculator"
    assert first["result"]["success"] is True