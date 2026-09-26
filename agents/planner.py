from typing import Any, Dict, List
import re


class AgentPlanner:
    """
    Deterministic planner for AURA.

    Produces executable task graphs for:
      - RAG retrieval
      - evidence analysis
      - calculator/tool execution
      - multi-step analytical queries
    """

    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry

    # =========================================================
    # PUBLIC API
    # =========================================================

    def plan(self, query: str) -> Dict[str, Any]:
        query = str(query or "").strip()

        if not query:
            return {
                "query": query,
                "tasks": [],
                "planning": {
                    "strategy": "empty_query",
                    "requires_analysis": False,
                    "task_count": 0,
                    "tools": [],
                },
            }

        q = query.lower()

        # -----------------------------------------------------
        # Arithmetic/tool query
        # -----------------------------------------------------

        if self._is_arithmetic_query(q):
            return self._build_calculator_plan(query)

        # -----------------------------------------------------
        # Normal RAG / analytical query
        # -----------------------------------------------------

        tasks = [
            {
                "id": 1,
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
        ]

        requires_analysis = self._requires_analysis(q)

        if requires_analysis:
            tasks.append(
                {
                    "id": 2,
                    "description": (
                        "Analyze the retrieved evidence "
                        "and identify the main supported points."
                    ),
                    "tool": "evidence_analyzer",
                    "arguments": {
                        "mode": "analyze",
                    },
                    "depends_on": [1],
                }
            )

        tasks = self._validate_tasks(tasks)

        return {
            "query": query,
            "tasks": tasks,
            "planning": {
                "strategy": "dynamic_rule_based",
                "requires_analysis": requires_analysis,
                "task_count": len(tasks),
                "tools": [
                    task["tool"]
                    for task in tasks
                ],
            },
        }

    # Compatibility APIs
    def create_plan(self, query: str) -> Dict[str, Any]:
        return self.plan(query)

    def build_plan(self, query: str) -> Dict[str, Any]:
        return self.plan(query)

    def generate_plan(self, query: str) -> Dict[str, Any]:
        return self.plan(query)

    # =========================================================
    # ANALYSIS DETECTION
    # =========================================================

    def _requires_analysis(self, q: str) -> bool:
        analysis_terms = (
            "analyze",
            "analyse",
            "analysis",
            "architecture",
            "architectural",
            "component",
            "components",
            "design",
            "compare",
            "comparison",
            "evaluate",
            "evaluation",
            "explain",
            "explanation",
            "detailed",
            "detail",
            "production",
            "deployment",
            "infrastructure",
            "monitoring",
            "scaling",
            "security",
            "performance",
            "strategy",
            "requirements",
            "tradeoff",
            "trade-offs",
        )

        return any(
            re.search(
                rf"\b{re.escape(term)}\b",
                q,
            )
            for term in analysis_terms
        )

    # =========================================================
    # ARITHMETIC DETECTION
    # =========================================================

    def _is_arithmetic_query(self, q: str) -> bool:

        arithmetic_words = (
            "calculate",
            "compute",
            "solve",
            "evaluate",
        )

        if any(
            word in q
            for word in arithmetic_words
        ):
            return bool(
                re.search(
                    r"\d",
                    q,
                )
            )

        return bool(
            re.fullmatch(
                r"[\d\s\+\-\*\/\(\)\.%]+",
                q,
            )
        )

    def _build_calculator_plan(
        self,
        query: str,
    ) -> Dict[str, Any]:

        task = {
            "id": 1,
            "description": "Calculate the requested expression.",
            "tool": "calculator",
            "arguments": {
                "expression": self._extract_expression(query),
            },
        }

        task = self._normalize_task(
            task,
            1,
        )

        return {
            "query": query,
            "tasks": [task],
            "planning": {
                "strategy": "arithmetic",
                "requires_analysis": False,
                "task_count": 1,
                "tools": ["calculator"],
            },
        }

    def _extract_expression(
        self,
        query: str,
    ) -> str:

        match = re.search(
            r"[\d\s\+\-\*\/\(\)\.%]+",
            query,
        )

        if match:
            return match.group(0).strip()

        return query

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_tasks(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        valid = []
        used_ids = set()

        for task in tasks:

            if not isinstance(task, dict):
                continue

            tool = task.get("tool")

            if not tool:
                continue

            task_id = task.get("id")

            if not isinstance(
                task_id,
                int,
            ):
                task_id = len(valid) + 1

            while task_id in used_ids:
                task_id += 1

            used_ids.add(task_id)

            normalized = self._normalize_task(
                task,
                task_id,
            )

            valid.append(normalized)

        return valid

    def _normalize_task(
        self,
        task: Dict[str, Any],
        task_id: int,
    ) -> Dict[str, Any]:

        normalized = {
            "id": task_id,
            "description": task.get(
                "description",
                "",
            ),
            "tool": task.get(
                "tool",
            ),
            "arguments": task.get(
                "arguments",
                {},
            ),
        }

        if not isinstance(
            normalized["arguments"],
            dict,
        ):
            normalized["arguments"] = {}

        dependencies = task.get(
            "depends_on",
        )

        if dependencies is not None:

            if isinstance(
                dependencies,
                int,
            ):
                dependencies = [
                    dependencies
                ]

            if isinstance(
                dependencies,
                list,
            ):
                normalized[
                    "depends_on"
                ] = dependencies

        return normalized


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

Planner = AgentPlanner