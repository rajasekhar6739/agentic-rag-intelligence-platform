from typing import Any


class ToolExecutor:
    """
    Executes registered AURA tools safely and returns
    a standardized execution result.
    """

    def __init__(self, registry):
        self.registry = registry

    # =========================================================
    # EXECUTE
    # =========================================================

    def execute(
        self,
        tool_name: str,
        arguments: dict | None = None
    ) -> dict:

        arguments = arguments or {}

        # -----------------------------------------------------
        # Check tool
        # -----------------------------------------------------

        if not self.registry.has(
            tool_name
        ):

            return {
                "success": False,
                "tool": tool_name,
                "arguments": arguments,
                "result": None,
                "error": (
                    f"Tool '{tool_name}' "
                    "is not registered."
                )
            }

        # -----------------------------------------------------
        # Execute
        # -----------------------------------------------------

        try:

            result = self.registry.execute(
                tool_name,
                **arguments
            )

            return {
                "success": True,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "error": None
            }

        except Exception as exc:

            return {
                "success": False,
                "tool": tool_name,
                "arguments": arguments,
                "result": None,
                "error": str(exc)
            }