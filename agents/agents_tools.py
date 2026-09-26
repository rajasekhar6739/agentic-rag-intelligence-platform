from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.rag_tool import RAGSearchTool


class AgentTools:

    def __init__(self, retriever):
        self.registry = ToolRegistry()

        self.registry.register(
            CalculatorTool()
        )

        self.registry.register(
            RAGSearchTool(retriever)
        )

    def execute(self, name, arguments):
        return self.registry.execute(
            name,
            arguments
        )

    def names(self):
        return self.registry.list_tools()