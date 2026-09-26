from __future__ import annotations

from typing import Any, Dict, List, Optional


class ToolRegistry:
    """
    Central registry for AURA tools.

    Built-in tools:
        - calculator
        - rag_search
        - evidence_analyzer
    """

    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.rag_tool = None

        self._register_default_tools()

    # ============================================================
    # REGISTRATION
    # ============================================================

    def register(
        self,
        name: str,
        tool: Any,
        description: Optional[str] = None,
    ) -> None:

        if not name:
            raise ValueError(
                "Tool name cannot be empty."
            )

        name = str(name)

        if description is not None:
            try:
                setattr(
                    tool,
                    "description",
                    description,
                )
            except Exception:
                pass

        self.tools[name] = tool

        if name == "rag_search":
            self.rag_tool = tool

    def unregister(
        self,
        name: str,
    ) -> bool:

        if name not in self.tools:
            return False

        del self.tools[name]

        if name == "rag_search":
            self.rag_tool = None

        return True

    # ============================================================
    # LOOKUP
    # ============================================================

    def get(
        self,
        name: str,
    ) -> Any:

        return self.tools.get(name)

    def has(
        self,
        name: str,
    ) -> bool:

        return name in self.tools

    # ============================================================
    # TOOL LISTS
    # ============================================================

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def names(self) -> List[str]:
        return self.list_tools()

    def available_tools(self) -> List[str]:
        return self.list_tools()

    # ============================================================
    # DESCRIPTIONS
    # ============================================================

    def descriptions(self) -> Dict[str, str]:

        result = {}

        for name, tool in self.tools.items():

            description = getattr(
                tool,
                "description",
                None,
            )

            if description is None:
                description = getattr(
                    tool,
                    "__doc__",
                    "",
                )

            result[name] = str(
                description or ""
            ).strip()

        return result

    def describe(self):

        result = {}

        for name, tool in self.tools.items():

            description = getattr(
                tool,
                "description",
                None,
            )

            if description is None:
                description = getattr(
                    tool,
                    "__doc__",
                    "",
                )

            result[name] = {
                "name": name,
                "description": str(
                    description or ""
                ).strip(),
                "type": type(tool).__name__,
            }

        return result

    # ============================================================
    # DEFAULT TOOLS
    # ============================================================

    def _register_default_tools(self):

        # --------------------------------------------------------
        # CALCULATOR
        # --------------------------------------------------------

        try:

            from tools.calculator import Calculator

            calculator = Calculator()

            self.register(
                "calculator",
                calculator,
            )

        except Exception:

            try:

                from tools.calculator_tool import CalculatorTool

                calculator = CalculatorTool()

                self.register(
                    "calculator",
                    calculator,
                )

            except Exception:
                pass

        # --------------------------------------------------------
        # RAG
        # --------------------------------------------------------

        try:

            from tools.rag_tool import RAGTool

            rag = RAGTool()

            self.register(
                "rag_search",
                rag,
            )

        except Exception as exc:

            print(
                "WARNING: RAG tool registration failed:",
                repr(exc),
            )

        # --------------------------------------------------------
        # EVIDENCE ANALYZER
        # --------------------------------------------------------

        try:

            from agents.evidence_analyzer import EvidenceAnalyzer

            analyzer = EvidenceAnalyzer()

            self.register(
                "evidence_analyzer",
                analyzer,
            )

        except Exception:

            try:

                from tools.evidence_analyzer import EvidenceAnalyzer

                analyzer = EvidenceAnalyzer()

                self.register(
                    "evidence_analyzer",
                    analyzer,
                )

            except Exception:
                pass

    # ============================================================
    # DOCUMENT LOADING
    # ============================================================

    def load_documents(
        self,
        documents: Any,
    ):

        rag = self.get(
            "rag_search"
        )

        if rag is None:
            raise RuntimeError(
                "rag_search tool is not registered."
            )

        loader = getattr(
            rag,
            "load_documents",
            None,
        )

        if loader is None:
            raise AttributeError(
                "Registered rag_search tool "
                "does not support document loading."
            )

        return loader(
            documents
        )

    # ============================================================
    # RAG SEARCH
    # ============================================================

    def rag_search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:

        tool = self.get(
            "rag_search"
        )

        if tool is None:

            return {
                "success": False,
                "query": query,
                "results": [],
                "error": (
                    "rag_search tool is not registered."
                ),
            }

        arguments = {
            "query": query,
            "top_k": top_k,
            **kwargs,
        }

        return self._invoke_tool(
            "rag_search",
            tool,
            arguments,
        )

    # ============================================================
    # CALCULATOR
    # ============================================================

    def calculator(
        self,
        expression: str,
        **kwargs,
    ) -> Dict[str, Any]:

        tool = self.get(
            "calculator"
        )

        if tool is None:

            return {
                "success": False,
                "tool": "calculator",
                "error": (
                    "calculator tool is not registered."
                ),
            }

        arguments = {
            "expression": expression,
            **kwargs,
        }

        return self._invoke_tool(
            "calculator",
            tool,
            arguments,
        )

    # ============================================================
    # EVIDENCE ANALYZER
    # ============================================================

    def evidence_analyzer(
        self,
        mode: str = "analyze",
        evidence=None,
        **kwargs,
    ) -> Dict[str, Any]:

        tool = self.get(
            "evidence_analyzer"
        )

        if tool is None:

            return {
                "success": False,
                "error": (
                    "evidence_analyzer "
                    "is not registered."
                ),
            }

        arguments = {
            "mode": mode,
            "evidence": evidence,
            **kwargs,
        }

        return self._invoke_tool(
            "evidence_analyzer",
            tool,
            arguments,
        )

    # ============================================================
    # GENERIC EXECUTION
    # ============================================================

    def execute(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        tool = self.get(name)

        if tool is None:

            return {
                "success": False,
                "tool": name,
                "error": (
                    f"Tool '{name}' is not registered."
                ),
            }

        arguments = arguments or {}

        return self._invoke_tool(
            name,
            tool,
            arguments,
        )

    # ============================================================
    # TOOL INVOCATION
    # ============================================================

    def _invoke_tool(
        self,
        name: str,
        tool: Any,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        try:

            if hasattr(tool, "run"):

                result = tool.run(
                    **arguments
                )

            elif hasattr(tool, "execute"):

                result = tool.execute(
                    **arguments
                )

            elif hasattr(tool, "search") and name == "rag_search":

                result = tool.search(
                    arguments.get("query", ""),
                    top_k=arguments.get(
                        "top_k",
                        5,
                    ),
                )

            elif callable(tool):

                result = tool(
                    **arguments
                )

            else:

                raise TypeError(
                    f"Tool '{name}' is not executable."
                )

            # RAG search() returns a list.
            if name == "rag_search":

                if isinstance(result, list):

                    return {
                        "success": True,
                        "tool": name,
                        "query": arguments.get(
                            "query",
                            "",
                        ),
                        "results": result,
                    }

                if isinstance(result, dict):

                    if "results" not in result:
                        result = {
                            **result,
                            "results": [],
                        }

                    result.setdefault(
                        "success",
                        True,
                    )

                    result.setdefault(
                        "tool",
                        name,
                    )

                    return result

            return {
                "success": True,
                "tool": name,
                "result": result,
            }

        except Exception as exc:

            return {
                "success": False,
                "tool": name,
                "error": str(exc),
            }

    # ============================================================
    # REPRESENTATION
    # ============================================================

    def __repr__(self):

        return (
            f"ToolRegistry("
            f"tools={self.list_tools()}"
            f")"
        )