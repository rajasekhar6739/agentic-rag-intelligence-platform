
from __future__ import annotations

import time
from typing import Any, Dict


class ToolExecutionError(Exception):
    """Raised when a tool cannot be executed."""


class ToolExecutor:
    """
    Central tool execution layer.

    Supports:
    - tool discovery
    - tool execution
    - retries
    - structured results
    - execution history
    - statistics
    """

    def __init__(
        self,
        tool_registry=None,
        max_retries: int = 2,
        retry_delay: float = 0.0,
    ):
        self.tool_registry = tool_registry
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # TOOL DISCOVERY
    # ------------------------------------------------------------------

    def available_tools(self) -> list[str]:
        """
        Return names of all tools known by the registry.
        """
        registry = self.tool_registry

        if registry is None:
            return []

        # Common registry APIs
        for method_name in (
            "available_tools",
            "list_tools",
            "get_tool_names",
            "names",
        ):
            method = getattr(registry, method_name, None)

            if callable(method):
                try:
                    result = method()

                    if isinstance(result, dict):
                        return [str(k) for k in result.keys()]

                    if isinstance(result, (list, tuple, set)):
                        return [str(x) for x in result]

                except Exception:
                    pass

        # Common registry attributes
        for attr_name in (
            "tools",
            "_tools",
            "registry",
            "_registry",
        ):
            value = getattr(registry, attr_name, None)

            if isinstance(value, dict):
                return [str(k) for k in value.keys()]

            if isinstance(value, (list, tuple, set)):
                return [str(x) for x in value]

        return []

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_name: str,
        arguments: dict | None = None,
        context: dict | None = None,
        max_retries: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:

        tool_name = str(tool_name).strip()

        if not tool_name:
            return self._failure(
                tool_name="",
                arguments=arguments or {},
                error="Tool name is required.",
            )

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            return self._failure(
                tool_name=tool_name,
                arguments={},
                error="Tool arguments must be a dictionary.",
            )

        retries = (
            self.max_retries
            if max_retries is None
            else max(0, int(max_retries))
        )

        last_error = None
        started = time.perf_counter()

        for attempt in range(1, retries + 2):
            try:
                result = self._call_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    context=context,
                    **kwargs,
                )

                elapsed = time.perf_counter() - started

                normalized = self._normalize_result(
                    tool_name=tool_name,
                    result=result,
                    attempt=attempt,
                    elapsed=elapsed,
                )

                self.history.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "attempt": attempt,
                        "success": normalized["success"],
                        "elapsed": elapsed,
                    }
                )

                return normalized

            except Exception as exc:
                last_error = str(exc)

                if attempt <= retries and self.retry_delay > 0:
                    time.sleep(self.retry_delay)

        elapsed = time.perf_counter() - started

        result = {
            "success": False,
            "tool": tool_name,
            "arguments": arguments,
            "attempts": retries + 1,
            "elapsed": elapsed,
            "error": last_error or "Tool execution failed.",
        }

        self.history.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "attempt": retries + 1,
                "success": False,
                "elapsed": elapsed,
                "error": result["error"],
            }
        )

        return result

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return self.execute(tool_name, arguments, **kwargs)

    def run(
        self,
        tool_name: str,
        arguments: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return self.execute(tool_name, arguments, **kwargs)

    # ------------------------------------------------------------------
    # INTERNAL TOOL CALLING
    # ------------------------------------------------------------------

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict,
        context: dict | None = None,
        **kwargs,
    ) -> Any:

        registry = self.tool_registry

        if registry is None:
            raise ToolExecutionError(
                f"No tool registry configured for '{tool_name}'."
            )

        # Registry-level execute()
        execute_method = getattr(registry, "execute", None)

        if callable(execute_method):
            attempts = [
                lambda: execute_method(
                    tool_name,
                    arguments,
                    context=context,
                ),
                lambda: execute_method(
                    tool_name,
                    arguments,
                ),
                lambda: execute_method(
                    name=tool_name,
                    arguments=arguments,
                    context=context,
                ),
                lambda: execute_method(
                    tool_name=tool_name,
                    arguments=arguments,
                ),
            ]

            for call in attempts:
                try:
                    return call()
                except TypeError:
                    continue

        # Registry-level get()
        get_method = getattr(registry, "get", None)

        if callable(get_method):
            tool = get_method(tool_name)

            if tool is None:
                raise ToolExecutionError(
                    f"Tool '{tool_name}' not found."
                )

            return self._invoke_tool(
                tool,
                arguments,
                context=context,
                **kwargs,
            )

        # Dictionary registry
        if isinstance(registry, dict):
            if tool_name not in registry:
                raise ToolExecutionError(
                    f"Tool '{tool_name}' not found."
                )

            return self._invoke_tool(
                registry[tool_name],
                arguments,
                context=context,
                **kwargs,
            )

        raise ToolExecutionError(
            f"Unable to execute tool '{tool_name}'."
        )

    def _invoke_tool(
        self,
        tool: Any,
        arguments: dict,
        context: dict | None = None,
        **kwargs,
    ) -> Any:

        if callable(tool):
            try:
                return tool(arguments, context=context, **kwargs)
            except TypeError:
                try:
                    return tool(**arguments)
                except TypeError:
                    return tool(arguments)

        for method_name in (
            "execute",
            "run",
            "invoke",
            "call",
        ):
            method = getattr(tool, method_name, None)

            if callable(method):
                try:
                    return method(arguments, context=context, **kwargs)
                except TypeError:
                    try:
                        return method(**arguments)
                    except TypeError:
                        return method(arguments)

        raise ToolExecutionError(
            f"Registered object for tool is not executable."
        )

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_result(
        self,
        tool_name: str,
        result: Any,
        attempt: int,
        elapsed: float,
    ) -> dict[str, Any]:

        if isinstance(result, dict):
            normalized = dict(result)
        else:
            normalized = {
                "result": result,
            }

        if "success" not in normalized:
            normalized["success"] = True

        normalized["tool"] = tool_name
        normalized["attempts"] = attempt
        normalized["elapsed"] = elapsed

        return normalized

    def _failure(
        self,
        tool_name: str,
        arguments: dict,
        error: str,
    ) -> dict[str, Any]:

        result = {
            "success": False,
            "tool": tool_name,
            "arguments": arguments,
            "attempts": 0,
            "elapsed": 0.0,
            "error": error,
        }

        self.history.append(result)

        return result

    # ------------------------------------------------------------------
    # HISTORY / METRICS
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    def clear_history(self) -> None:
        self.history.clear()

    def stats(self) -> dict[str, Any]:
        total = len(self.history)
        successful = sum(
            1 for item in self.history
            if item.get("success") is True
        )
        failed = total - successful

        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (
                successful / total
                if total
                else 0.0
            ),
            "available_tools": self.available_tools(),
        }