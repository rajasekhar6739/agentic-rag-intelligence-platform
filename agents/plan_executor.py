from agents.agent_tools import AgentTools


class PlanExecutor:

    def __init__(self, retriever):

        self.tools = AgentTools(
            retriever
        )

    def execute(self, plan):

        results = []

        for task in plan.get(
            "tasks",
            []
        ):

            tool_name = task.get(
                "tool"
            )

            description = task.get(
                "description",
                ""
            )

            arguments = self._arguments(
                tool_name,
                description
            )

            result = self.tools.execute(
                tool_name,
                arguments
            )

            results.append({
                "task_id": task.get("id"),
                "tool": tool_name,
                "description": description,
                "arguments": arguments,
                "result": result
            })

        return results

    def _arguments(
        self,
        tool_name,
        description
    ):

        if tool_name == "calculator":

            return {
                "expression": description
            }

        if tool_name == "rag_search":

            return {
                "query": description,
                "top_k": 5
            }

        return {}