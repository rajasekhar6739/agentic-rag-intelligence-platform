from __future__ import annotations

from typing import Any


class ResultSynthesizer:
    """
    Final answer synthesis layer for AURA.

    Compatible with:
      synthesizer.synthesize(
          query=...,
          evidence=...,
          results=...
      )
    """

    def __init__(self, *args, **kwargs):
        self.llm = kwargs.get("llm")
        self.model = kwargs.get("model")

    # ============================================================
    # PUBLIC API
    # ============================================================

    def synthesize(
        self,
        query: str | None = None,
        evidence: Any = None,
        results: Any = None,
        answer: str | None = None,
        **kwargs,
    ):
        """
        Synthesize a grounded answer.

        `results` is accepted as a compatibility alias for
        `evidence`.
        """

        if evidence is None:
            evidence = results

        if evidence is None:
            evidence = []

        normalized = self._normalize_evidence(evidence)

        if not normalized:
            return {
                "success": False,
                "answer": (
                    "The provided evidence is insufficient "
                    "to answer the question."
                ),
                "grounded": False,
                "confidence": 0.0,
                "citations": [],
                "evidence_count": 0,
            }

        # If an answer was explicitly supplied, preserve it.
        if answer:
            final_answer = str(answer)
        else:
            final_answer = self._build_answer(
                query or "",
                normalized,
            )

        citations = list(
            range(1, len(normalized) + 1)
        )

        confidence = self._confidence(
            normalized
        )

        return {
            "success": True,
            "answer": final_answer,
            "grounded": True,
            "confidence": confidence,
            "citations": citations,
            "evidence_count": len(normalized),
            "evidence": normalized,
        }

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_evidence(self, evidence):

        if evidence is None:
            return []

        # Direct list.
        if isinstance(evidence, list):
            output = []

            for item in evidence:
                normalized = self._normalize_item(item)

                if normalized:
                    output.append(normalized)

            return output

        # Dict containing evidence/results.
        if isinstance(evidence, dict):

            for key in (
                "evidence",
                "results",
                "documents",
                "items",
            ):
                if key in evidence:
                    return self._normalize_evidence(
                        evidence[key]
                    )

            item = self._normalize_item(
                evidence
            )

            return [item] if item else []

        # String evidence.
        if isinstance(evidence, str):
            text = evidence.strip()

            if not text:
                return []

            return [
                {
                    "text": text,
                    "source": "unknown",
                    "page": None,
                }
            ]

        return []

    def _normalize_item(self, item):

        if item is None:
            return None

        if isinstance(item, str):
            text = item.strip()

            if not text:
                return None

            return {
                "text": text,
                "source": "unknown",
                "page": None,
            }

        if isinstance(item, dict):

            text = (
                item.get("text")
                or item.get("content")
                or item.get("answer")
                or item.get("description")
                or ""
            )

            text = str(text).strip()

            if not text:
                return None

            return {
                "text": text,
                "source": item.get(
                    "source",
                    item.get("file", "unknown"),
                ),
                "page": item.get("page"),
                "score": item.get("score"),
                "chunk_id": item.get("chunk_id"),
            }

        return None

    # ============================================================
    # ANSWER BUILDING
    # ============================================================

    def _build_answer(
        self,
        query: str,
        evidence: list[dict],
    ):

        parts = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            text = item["text"].strip()

            if not text:
                continue

            parts.append(
                f"{text} [{index}]"
            )

        if not parts:
            return (
                "The provided evidence is "
                "insufficient to answer the question."
            )

        return " ".join(parts)

    # ============================================================
    # CONFIDENCE
    # ============================================================

    def _confidence(self, evidence):

        if not evidence:
            return 0.0

        scores = []

        for item in evidence:
            score = item.get("score")

            if isinstance(score, (int, float)):
                scores.append(float(score))

        if scores:
            value = sum(scores) / len(scores)

            # Normalize common retrieval score ranges.
            if value > 1:
                value = 1.0

            if value < 0:
                value = 0.0

            return round(value, 2)

        return 0.75

    # ============================================================
    # ALIASES
    # ============================================================

    def generate(
        self,
        query=None,
        evidence=None,
        results=None,
        **kwargs,
    ):
        return self.synthesize(
            query=query,
            evidence=evidence,
            results=results,
            **kwargs,
        )

    def build(
        self,
        query=None,
        evidence=None,
        results=None,
        **kwargs,
    ):
        return self.synthesize(
            query=query,
            evidence=evidence,
            results=results,
            **kwargs,
        )