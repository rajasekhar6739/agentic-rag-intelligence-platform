from __future__ import annotations

from typing import Any, Dict, Optional


class PlannedAgent:
    """
    AURA Planned Agent.

    Supports direct retriever injection:

        PlannedAgent(FakeRetriever())

    The agent performs lightweight planning and executes:
        - rag_search
        - calculator

    This implementation is intentionally compatible with the
    current test_planned_agent.py contract.
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
    ):
        self.retriever = retriever
        self.orchestrator = orchestrator

    # ============================================================
    # DOCUMENT MANAGEMENT
    # ============================================================

    def load_documents(
        self,
        documents: Any,
    ) -> Any:
        """
        Load documents into the injected retriever when supported.
        """

        if self.retriever is None:
            raise RuntimeError(
                "Retriever is not configured."
            )

        # Existing retriever/index implementations may expose build().
        if hasattr(
            self.retriever,
            "build",
        ):
            normalized = []

            for index, document in enumerate(
                documents
            ):

                if isinstance(
                    document,
                    str,
                ):
                    normalized.append(
                        {
                            "text": document,
                            "source": None,
                            "page": None,
                            "chunk_id": index,
                        }
                    )

                elif isinstance(
                    document,
                    dict,
                ):
                    normalized.append(
                        {
                            "text": document["text"],
                            "source": document.get(
                                "source"
                            ),
                            "page": document.get(
                                "page"
                            ),
                            "chunk_id": document.get(
                                "chunk_id",
                                index,
                            ),
                        }
                    )

                elif hasattr(
                    document,
                    "text",
                ):
                    normalized.append(
                        {
                            "text": document.text,
                            "source": getattr(
                                document,
                                "source",
                                None,
                            ),
                            "page": getattr(
                                document,
                                "page",
                                None,
                            ),
                            "chunk_id": getattr(
                                document,
                                "chunk_id",
                                index,
                            ),
                        }
                    )

                else:
                    raise TypeError(
                        "Unsupported document type."
                    )

            self.retriever.build(
                normalized
            )

            return {
                "success": True,
                "documents_loaded": len(
                    normalized
                ),
            }

        raise AttributeError(
            "Configured retriever does not "
            "support document loading."
        )

    # ============================================================
    # PLANNING
    # ============================================================

    def _build_plan(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Build a simple deterministic plan.

        The current test query contains both:
            - a RAG/document question
            - an arithmetic calculation
        """

        tasks = []

        lowered = query.lower()

        # --------------------------------------------------------
        # RAG task
        # --------------------------------------------------------

        rag_keywords = [
            "document",
            "documents",
            "rag",
            "knowledge",
            "according to",
            "source",
            "evidence",
            "explain",
        ]

        needs_rag = any(
            keyword in lowered
            for keyword in rag_keywords
        )

        if needs_rag:
            tasks.append(
                {
                    "id": len(tasks) + 1,
                    "description": (
                        "Retrieve relevant evidence "
                        "from the knowledge base."
                    ),
                    "tool": "rag_search",
                    "arguments": {
                        "query": query,
                        "top_k": 5,
                    },
                }
            )

        # --------------------------------------------------------
        # Calculator task
        # --------------------------------------------------------

        arithmetic_keywords = [
            "calculate",
            "percent",
            "%",
            "*",
            "/",
            "+",
            "-",
        ]

        needs_calculator = any(
            keyword in lowered
            for keyword in arithmetic_keywords
        )

        if needs_calculator:
            expression = self._extract_expression(
                query
            )

            if expression:
                tasks.append(
                    {
                        "id": len(tasks) + 1,
                        "description": (
                            "Calculate the requested "
                            "arithmetic expression."
                        ),
                        "tool": "calculator",
                        "arguments": {
                            "expression": expression,
                        },
                    }
                )

        return {
            "query": query,
            "tasks": tasks,
            "planning": {
                "strategy": "dynamic_rule_based",
                "requires_analysis": (
                    len(tasks) > 1
                ),
                "task_count": len(tasks),
                "tools": [
                    task["tool"]
                    for task in tasks
                ],
            },
        }

    # ============================================================
    # EXPRESSION EXTRACTION
    # ============================================================

    def _extract_expression(
        self,
        query: str,
    ) -> Optional[str]:
        """
        Extract the arithmetic expression.

        Supports examples such as:

            125 * 8 + 50 / 2

            250 * 18 percent

            calculate 100 + 20
        """

        import re

        text = query.strip()

        # --------------------------------------------------------
        # Percentage pattern
        #
        # "250 * 18 percent"
        # becomes:
        #
        # 250 * 18 / 100
        # --------------------------------------------------------

        percent_match = re.search(
            r"(\d+(?:\.\d+)?)\s*"
            r"\*\s*"
            r"(\d+(?:\.\d+)?)\s*"
            r"percent",
            text,
            re.IGNORECASE,
        )

        if percent_match:

            first = percent_match.group(
                1
            )

            second = percent_match.group(
                2
            )

            return (
                f"{first} * {second} / 100"
            )

        # --------------------------------------------------------
        # Explicit % notation
        #
        # "250 * 18%"
        # --------------------------------------------------------

        percent_symbol = re.search(
            r"(\d+(?:\.\d+)?)\s*"
            r"\*\s*"
            r"(\d+(?:\.\d+)?)\s*%",
            text,
        )

        if percent_symbol:

            first = percent_symbol.group(
                1
            )

            second = percent_symbol.group(
                2
            )

            return (
                f"{first} * {second} / 100"
            )

        # --------------------------------------------------------
        # Normal arithmetic expression
        # --------------------------------------------------------

        matches = re.findall(
            r"[0-9+\-*/().\s]+",
            text,
        )

        if not matches:
            return None

        candidates = [
            value.strip()
            for value in matches
            if any(
                char in value
                for char in "+-*/"
            )
        ]

        if not candidates:
            return None

        # Choose the longest arithmetic expression.
        return max(
            candidates,
            key=len,
        ).strip()

    # ============================================================
    # RAG SEARCH
    # ============================================================

    def _rag_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute retrieval against the injected retriever.
        """

        if self.retriever is None:
            return {
                "success": False,
                "results": [],
                "error": (
                    "Retriever is not configured."
                ),
            }

        if not hasattr(
            self.retriever,
            "search",
        ):
            return {
                "success": False,
                "results": [],
                "error": (
                    "Retriever does not "
                    "support search()."
                ),
            }

        # FakeRetriever uses top_k.
        try:
            results = self.retriever.search(
                query,
                top_k=top_k,
            )

        except TypeError:
            # HybridRetriever in this project uses k.
            results = self.retriever.search(
                query,
                k=top_k,
            )

        normalized = []

        for item in results or []:

            if not isinstance(
                item,
                dict,
            ):
                continue

            # FakeRetriever returns:
            #
            # {
            #   "document": {...},
            #   "rerank_score": 4.5
            # }
            #
            # Other retrievers may directly return the
            # document dictionary.

            document = item.get(
                "document",
                item,
            )

            if not isinstance(
                document,
                dict,
            ):
                continue

            normalized.append(
                {
                    "text": document.get(
                        "text",
                        "",
                    ),
                    "source": document.get(
                        "source"
                    ),
                    "page": document.get(
                        "page"
                    ),
                    "chunk_id": document.get(
                        "chunk_id"
                    ),
                    "rerank_score": item.get(
                        "rerank_score",
                        item.get(
                            "score",
                            0.0,
                        ),
                    ),
                }
            )

        return {
            "success": True,
            "query": query,
            "results": normalized,
        }

    # ============================================================
    # CALCULATOR
    # ============================================================

    def _calculator(
        self,
        expression: str,
    ) -> Dict[str, Any]:
        """
        Safe arithmetic calculator.

        Only numeric arithmetic characters are allowed.
        """

        import ast
        import operator

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def evaluate(
            node,
        ):
            if isinstance(
                node,
                ast.Expression,
            ):
                return evaluate(
                    node.body
                )

            if isinstance(
                node,
                ast.Constant,
            ):
                if isinstance(
                    node.value,
                    (int, float),
                ):
                    return node.value

                raise ValueError(
                    "Invalid numeric value."
                )

            if isinstance(
                node,
                ast.BinOp,
            ):
                left = evaluate(
                    node.left
                )

                right = evaluate(
                    node.right
                )

                operation = operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Unsupported operator."
                    )

                return operation(
                    left,
                    right,
                )

            if isinstance(
                node,
                ast.UnaryOp,
            ):
                value = evaluate(
                    node.operand
                )

                operation = operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Unsupported unary operator."
                    )

                return operation(
                    value
                )

            raise ValueError(
                "Invalid arithmetic expression."
            )

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )

            value = evaluate(tree)

            return {
                "success": True,
                "expression": expression,
                "result": value,
            }

        except Exception as exc:

            return {
                "success": False,
                "expression": expression,
                "result": None,
                "error": str(exc),
            }

    # ============================================================
    # RUN
    # ============================================================

    def run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the complete planned workflow.
        """

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------------
        # If a real orchestrator was explicitly supplied,
        # use it.
        # --------------------------------------------------------

        if (
            self.orchestrator is not None
            and hasattr(
                self.orchestrator,
                "run",
            )
        ):
            raw = self.orchestrator.run(
                query,
                **kwargs,
            )

            return self._normalize_external_result(
                query,
                raw,
            )

        # --------------------------------------------------------
        # Build plan
        # --------------------------------------------------------

        plan = self._build_plan(
            query
        )

        print(
            "\n========================================"
        )
        print(
            "PLANNED AGENT"
        )
        print(
            "========================================"
        )

        print(
            "QUERY:",
            query,
        )

        print(
            "\nPLAN:"
        )

        print(plan)

        # --------------------------------------------------------
        # Execute tasks
        # --------------------------------------------------------

        results = []

        tools_used = []

        evidence = []

        for task in plan["tasks"]:

            tool = task["tool"]

            arguments = task[
                "arguments"
            ]

            print(
                "\nEXECUTING:",
                task["id"],
            )

            print(
                "TOOL:",
                tool,
            )

            print(
                "ARGUMENTS:",
                arguments,
            )

            if tool == "rag_search":

                tool_result = (
                    self._rag_search(
                        query=arguments[
                            "query"
                        ],
                        top_k=arguments[
                            "top_k"
                        ],
                    )
                )

                for item in tool_result.get(
                    "results",
                    [],
                ):
                    evidence.append(
                        item
                    )

            elif tool == "calculator":

                tool_result = (
                    self._calculator(
                        arguments[
                            "expression"
                        ]
                    )
                )

            else:

                tool_result = {
                    "success": False,
                    "error": (
                        f"Unknown tool: {tool}"
                    ),
                }

            tools_used.append(
                tool
            )

            results.append(
                {
                    "task_id": task["id"],
                    "tool": tool,
                    "result": tool_result,
                }
            )

        # --------------------------------------------------------
        # Build final answer
        # --------------------------------------------------------

        answer_parts = []

        # RAG answer
        if evidence:

            answer_parts.append(
                "Based on the retrieved "
                "documents:"
            )

            for index, item in enumerate(
                evidence,
                start=1,
            ):

                source = item.get(
                    "source"
                )

                page = item.get(
                    "page"
                )

                citation = ""

                if source:
                    citation += (
                        f" [{source}"
                    )

                    if page is None:
                        citation += "]"
                    else:
                        citation += (
                            f", p. {page}]"
                        )

                answer_parts.append(
                    f"- {item.get('text', '')}"
                    f"{citation}"
                )

        # Calculator answer
        calculation = None

        for item in results:

            if item["tool"] != "calculator":
                continue

            calculation_result = item[
                "result"
            ]

            if calculation_result.get(
                "success"
            ):
                calculation = (
                    calculation_result[
                        "result"
                    ]
                )

        if calculation is not None:

            answer_parts.append(
                "\nCalculation: "
                f"{calculation}"
            )

        if not answer_parts:

            answer_parts.append(
                "The agent could not "
                "complete the requested task."
            )

        answer = "\n".join(
            answer_parts
        )

        grounded = bool(
            evidence
        )

        confidence = (
            0.9
            if evidence
            else 0.0
        )

        return {
            "success": True,
            "query": query,
            "answer": answer,
            "grounded": grounded,
            "confidence": confidence,
            "tools_used": tools_used,
            "plan": plan,
            "results": results,
            "evidence": evidence,
            "evidence_count": len(
                evidence
            ),
            "citations": list(
                range(
                    1,
                    len(evidence) + 1,
                )
            ),
            "critique": None,
            "attempt": 1,
            "attempts": [],
            "attempt_count": 1,
            "final_attempt": 1,
            "self_corrected": False,
        }

    # ============================================================
    # EXTERNAL RESULT NORMALIZATION
    # ============================================================

    def _normalize_external_result(
        self,
        query: str,
        result: Any,
    ) -> Dict[str, Any]:

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "Orchestrator must return "
                "a dictionary."
            )

        task_results = result.get(
            "results"
        )

        if task_results is None:

            graph = result.get(
                "task_graph",
                {},
            )

            if isinstance(
                graph,
                dict,
            ):
                graph_results = graph.get(
                    "results",
                    {},
                )

                if isinstance(
                    graph_results,
                    dict,
                ):
                    task_results = []

                    for task_id, value in (
                        graph_results.items()
                    ):

                        task_results.append(
                            {
                                "task_id": task_id,
                                "tool": value.get(
                                    "tool"
                                )
                                if isinstance(
                                    value,
                                    dict,
                                )
                                else None,
                                "result": value,
                            }
                        )

        if task_results is None:
            task_results = []

        return {
            **result,
            "query": query,
            "results": task_results,
            "tools_used": result.get(
                "tools_used",
                [],
            ),
            "answer": result.get(
                "answer",
                "",
            ),
            "grounded": result.get(
                "grounded",
                False,
            ),
            "confidence": result.get(
                "confidence",
                0.0,
            ),
            "plan": result.get(
                "plan"
            ),
        }

    # ============================================================
    # TOOL HELPERS
    # ============================================================

    def list_tools(self):

        tools = [
            "rag_search",
            "calculator",
        ]

        if self.orchestrator is not None:

            if hasattr(
                self.orchestrator,
                "list_tools",
            ):
                try:
                    return self.orchestrator.list_tools()
                except Exception:
                    pass

        return tools

    def __repr__(self):

        return (
            "PlannedAgent("
            f"retriever={type(self.retriever).__name__}"
            ")"
        )