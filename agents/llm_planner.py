from __future__ import annotations

import json
from typing import Any, Dict, List

from llm.client import LLMClient


class LLMPlanner:
    """
    LLM-driven planner for AURA.

    Converts a user request into a structured sequence of
    tool actions while validating requested tools against
    the registry.
    """

    SYSTEM_PROMPT = """
You are the planning engine of an AI agent.

Your job is to decide which tools are required to answer
the user's request.

Available tools:
- rag_search: search internal knowledge/documents
- calculator: perform mathematical calculations
- evidence_analyzer: analyze retrieved evidence

Return ONLY valid JSON.

Schema:

{
  "reasoning": "short explanation",
  "tool_calls": [
    {
      "tool": "tool_name",
      "arguments": {}
    }
  ]
}

Rules:
1. Use only available tools.
2. Do not invent tools.
3. If no tool is required, return an empty tool_calls list.
4. For internal-document questions, use rag_search.
5. For mathematical calculations, use calculator.
6. Use evidence_analyzer when retrieved evidence needs analysis.
7. Keep arguments explicit and minimal.
"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        available_tools: List[str] | None = None,
    ):
        self.llm = llm or LLMClient()

        self.available_tools = (
            available_tools
            or [
                "rag_search",
                "calculator",
                "evidence_analyzer",
            ]
        )

    def plan(
        self,
        request: str,
    ) -> Dict[str, Any]:

        if not isinstance(request, str):
            raise TypeError(
                "request must be a string."
            )

        request = request.strip()

        if not request:
            return {
                "success": False,
                "reasoning": "",
                "tool_calls": [],
                "error": "Request cannot be empty.",
            }

        prompt = self._build_prompt(request)

        result = self.llm.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=800,
        )

        if not result.get("success"):
            return self._fallback_plan(
                request
            )

        parsed = self._parse(
            result.get("text", "")
        )

        if not parsed.get("success"):
            return self._fallback_plan(
                request
            )

        return parsed

    def _build_prompt(
        self,
        request: str,
    ) -> str:

        return (
            "Available tools:\n"
            + json.dumps(
                self.available_tools,
                indent=2,
            )
            + "\n\nUser request:\n"
            + request
            + "\n\nReturn JSON only."
        )

    def _parse(
        self,
        text: str,
    ) -> Dict[str, Any]:

        if not text:
            return {
                "success": False,
                "tool_calls": [],
                "error": "Empty planner response.",
            }

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {
                "success": False,
                "tool_calls": [],
                "error": "Planner returned invalid JSON.",
            }

        if not isinstance(data, dict):
            return {
                "success": False,
                "tool_calls": [],
                "error": "Planner response must be an object.",
            }

        calls = data.get(
            "tool_calls",
            [],
        )

        if not isinstance(calls, list):
            calls = []

        validated = []

        for call in calls:

            if not isinstance(call, dict):
                continue

            tool = call.get("tool")

            if tool not in self.available_tools:
                continue

            arguments = call.get(
                "arguments",
                {},
            )

            if not isinstance(arguments, dict):
                arguments = {}

            validated.append(
                {
                    "tool": tool,
                    "arguments": arguments,
                }
            )

        return {
            "success": True,
            "reasoning": str(
                data.get(
                    "reasoning",
                    "",
                )
            ),
            "tool_calls": validated,
        }

    def _fallback_plan(
        self,
        request: str,
    ) -> Dict[str, Any]:

        text = request.lower()

        # Internal knowledge/document request
        rag_terms = [
            "internal",
            "document",
            "documents",
            "incident",
            "runbook",
            "production",
            "latency",
            "evidence",
            "knowledge base",
        ]

        if any(
            term in text
            for term in rag_terms
        ):
            return {
                "success": True,
                "reasoning": (
                    "Fallback planner selected "
                    "RAG for an internal knowledge request."
                ),
                "tool_calls": [
                    {
                        "tool": "rag_search",
                        "arguments": {
                            "query": request,
                            "top_k": 5,
                        },
                    }
                ],
            }

        # Calculator request
        calculation_terms = [
            "calculate",
            "multiply",
            "multiplied",
            "percentage",
            "percent",
            "divide",
            "plus",
            "minus",
            "subtract",
            "add",
        ]

        if any(
            term in text
            for term in calculation_terms
        ):
            return {
                "success": True,
                "reasoning": (
                    "Fallback planner selected "
                    "the calculator."
                ),
                "tool_calls": [
                    {
                        "tool": "calculator",
                        "arguments": {
                            "expression": request,
                        },
                    }
                ],
            }

        return {
            "success": True,
            "reasoning": (
                "No specialized tool was "
                "required."
            ),
            "tool_calls": [],
        }

    def __repr__(self) -> str:
        return (
            "LLMPlanner("
            f"tools={self.available_tools!r}"
            ")"
        )