from tools.calculator import Calculator
from tools.rag_tool import RAGSearchTool


class ToolRegistry:

    def __init__(self):

        self.calculator = Calculator()
        self.rag_tool = RAGSearchTool()

        self.tools = {
            "calculator": self.calculator,
            "rag_search": self.rag_tool
        }

    def load_documents(self, documents):

        self.rag_tool.load_documents(
            documents
        )

    def execute(
        self,
        tool_name: str,
        arguments: dict
    ) -> dict:

        if tool_name not in self.tools:

            return {
                "success": False,
                "tool": tool_name,
                "error": "Tool not found."
            }

        try:

            if tool_name == "calculator":

                result = self.calculator.calculate(
                    arguments["expression"]
                )

                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result
                }

            if tool_name == "rag_search":

                result = self.rag_tool.search(
                    query=arguments["query"],
                    top_k=arguments.get(
                        "top_k",
                        5
                    )
                )

                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result
                }

            return {
                "success": False,
                "tool": tool_name,
                "error": "Unsupported tool."
            }

        except Exception as exc:

            return {
                "success": False,
                "tool": tool_name,
                "error": str(exc)
            }