class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools.get(name)

    def list_tools(self):
        return list(self.tools.keys())

    def execute(self, name, arguments=None):

        tool = self.get(name)

        if tool is None:
            return {
                "success": False,
                "error": f"Unknown tool: {name}"
            }

        arguments = arguments or {}

        try:

            # New-style tools
            if hasattr(tool, "execute"):
                return tool.execute(arguments)

            # Existing/legacy tools
            if hasattr(tool, "run"):
                return tool.run(**arguments)

            return {
                "success": False,
                "error": (
                    f"Tool '{name}' has neither "
                    "'execute' nor 'run'."
                )
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc)
            }