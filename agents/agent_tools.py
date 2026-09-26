from __future__ import annotations

from typing import Any


class AgentTools:
    """
    Compatibility tool layer used by FunctionCaller and ToolAgent.
    """

    def __init__(self, tool_registry=None, tool_executor=None):
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor

        if self.tool_executor is None:
            try:
                from agents.tool_executor import ToolExecutor

                self.tool_executor = ToolExecutor(
                    tool_registry=tool_registry
                )
            except Exception:
                self.tool_executor = None

    def available_tools(self):
        if self.tool_executor is not None:
            fn = getattr(
                self.tool_executor,
                "available_tools",
                None,
            )
            if callable(fn):
                return fn()

        registry = self.tool_registry

        if registry is None:
            return []

        for name in (
            "available_tools",
            "list_tools",
            "get_tool_names",
            "names",
        ):
            fn = getattr(registry, name, None)
            if callable(fn):
                try:
                    result = fn()
                    if isinstance(result, dict):
                        return list(result.keys())
                    if isinstance(result, (list, tuple, set)):
                        return list(result)
                except Exception:
                    pass

        for attr in ("tools", "_tools", "registry", "_registry"):
            value = getattr(registry, attr, None)
            if isinstance(value, dict):
                return list(value.keys())

        return []

    def execute(
        self,
        tool_name: str,
        arguments: dict | None = None,
        **kwargs,
    ):
        arguments = arguments or {}

        if self.tool_executor is not None:
            return self.tool_executor.execute(
                tool_name=tool_name,
                arguments=arguments,
                **kwargs,
            )

        return {
            "success": False,
            "tool": tool_name,
            "arguments": arguments,
            "error": "Tool executor unavailable.",
        }

    def run(self, tool_name, arguments=None, **kwargs):
        return self.execute(
            tool_name,
            arguments,
            **kwargs,
        )

    def call(self, tool_name, arguments=None, **kwargs):
        return self.execute(
            tool_name,
            arguments,
            **kwargs,
        )

    def select(self, query: str):
        q = str(query).lower()

        if any(
            x in q
            for x in (
                "calculate",
                "compute",
                "solve",
                "evaluate",
            )
        ):
            return {
                "tool": "calculator",
                "arguments": {
                    "expression": query,
                },
            }

        if any(
            x in q
            for x in (
                "analyze",
                "architecture",
                "components",
                "compare",
            )
        ):
            return {
                "tool": "evidence_analyzer",
                "arguments": {
                    "query": query,
                    "mode": "analyze",
                },
            }

        return {
            "tool": "rag_search",
            "arguments": {
                "query": query,
                "top_k": 5,
            },
        }


class AgentToolSelector:
    """
    Existing selector API kept for compatibility.
    """

    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        self.tools = AgentTools(
            tool_registry=tool_registry
        )

    def select(self, query: str):
        return self.tools.select(query)

    def execute(self, query: str):
        selected = self.select(query)

        result = self.tools.execute(
            selected["tool"],
            selected.get("arguments", {}),
        )

        return {
            "selected_tools": [selected["tool"]],
            "results": [result],
        }


# Compatibility aliases
Tools = AgentTools
ToolManager = AgentTools