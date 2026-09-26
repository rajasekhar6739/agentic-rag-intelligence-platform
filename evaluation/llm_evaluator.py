from __future__ import annotations

from typing import Any, Dict, List


class LLMEvaluator:
    """
    Deterministic evaluation layer for the AI/RAG agent.

    Evaluates:
      - grounding
      - citation quality
      - answer relevance
      - tool usage
      - execution quality
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence: List[Any] | None = None,
        citations: List[str] | None = None,
        tools_used: List[str] | None = None,
    ) -> Dict[str, Any]:

        evidence = evidence or []
        citations = citations or []
        tools_used = tools_used or []

        question = str(question or "").strip()
        answer = str(answer or "").strip()

        grounding = self._grounding_score(
            answer,
            evidence,
        )

        relevance = self._relevance_score(
            question,
            answer,
        )

        citation_score = self._citation_score(
            evidence,
            citations,
        )

        tool_score = self._tool_score(
            tools_used,
        )

        overall = round(
            (
                grounding
                + relevance
                + citation_score
                + tool_score
            )
            / 4.0,
            3,
        )

        return {
            "success": True,
            "scores": {
                "grounding": grounding,
                "relevance": relevance,
                "citation_quality": citation_score,
                "tool_usage": tool_score,
                "overall": overall,
            },
            "grounded": grounding >= 0.5,
            "issues": self._issues(
                grounding,
                relevance,
                citation_score,
                tool_score,
            ),
        }

    # ---------------------------------------------------------
    # GROUNDING
    # ---------------------------------------------------------

    @staticmethod
    def _grounding_score(
        answer: str,
        evidence: List[Any],
    ) -> float:

        if not answer or not evidence:
            return 0.0

        answer_words = set(
            answer.lower().split()
        )

        evidence_words = set()

        for item in evidence:
            if isinstance(item, dict):
                text = item.get(
                    "text",
                    item.get("content", ""),
                )
            else:
                text = getattr(
                    item,
                    "text",
                    str(item),
                )

            evidence_words.update(
                str(text).lower().split()
            )

        if not answer_words:
            return 0.0

        overlap = (
            len(answer_words & evidence_words)
            / len(answer_words)
        )

        return round(
            min(1.0, overlap * 2),
            3,
        )

    # ---------------------------------------------------------
    # RELEVANCE
    # ---------------------------------------------------------

    @staticmethod
    def _relevance_score(
        question: str,
        answer: str,
    ) -> float:

        if not question or not answer:
            return 0.0

        question_words = set(
            question.lower().split()
        )

        answer_words = set(
            answer.lower().split()
        )

        if not question_words:
            return 0.0

        overlap = (
            len(question_words & answer_words)
            / len(question_words)
        )

        return round(
            min(1.0, overlap * 2),
            3,
        )

    # ---------------------------------------------------------
    # CITATIONS
    # ---------------------------------------------------------

    @staticmethod
    def _citation_score(
        evidence: List[Any],
        citations: List[str],
    ) -> float:

        if not evidence:
            return 1.0

        if not citations:
            return 0.0

        valid = 0

        for item in evidence:

            source = None
            page = None

            if isinstance(item, dict):
                source = item.get("source")
                page = item.get("page")
            else:
                source = getattr(
                    item,
                    "source",
                    None,
                )
                page = getattr(
                    item,
                    "page",
                    None,
                )

            if source:
                expected = f"[{source}"

                if any(
                    expected in citation
                    for citation in citations
                ):
                    valid += 1

        return round(
            valid / len(evidence),
            3,
        )

    # ---------------------------------------------------------
    # TOOL USAGE
    # ---------------------------------------------------------

    @staticmethod
    def _tool_score(
        tools_used: List[str],
    ) -> float:

        if not tools_used:
            return 0.5

        valid_tools = {
            "rag_search",
            "calculator",
            "evidence_analyzer",
            "multimodal",
            "pdf_loader",
        }

        valid = sum(
            1
            for tool in tools_used
            if tool in valid_tools
        )

        return round(
            min(
                1.0,
                valid / max(1, len(tools_used)),
            ),
            3,
        )

    # ---------------------------------------------------------
    # ISSUES
    # ---------------------------------------------------------

    @staticmethod
    def _issues(
        grounding: float,
        relevance: float,
        citation_score: float,
        tool_score: float,
    ) -> List[str]:

        issues = []

        if grounding < 0.5:
            issues.append(
                "Answer is weakly grounded in evidence."
            )

        if relevance < 0.5:
            issues.append(
                "Answer may not directly address the question."
            )

        if citation_score < 0.5:
            issues.append(
                "Citation coverage is incomplete."
            )

        if tool_score < 0.5:
            issues.append(
                "Tool usage could not be fully validated."
            )

        return issues