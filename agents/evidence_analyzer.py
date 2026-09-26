from typing import Any


class EvidenceAnalyzer:

    def __init__(self):
        pass

    def analyze(
        self,
        evidence=None,
        query: str = "",
        mode: str = "analyze",
        **kwargs,
    ) -> dict[str, Any]:

        if evidence is None:
            evidence = []

        # Accept dependency/reference formats too.
        if isinstance(evidence, dict):

            if "evidence" in evidence:
                evidence = evidence["evidence"]

            elif "results" in evidence:
                evidence = evidence["results"]

        if not isinstance(evidence, list):
            evidence = [evidence]

        normalized = []

        for item in evidence:

            if isinstance(item, dict):

                text = (
                    item.get("text")
                    or item.get("content")
                    or ""
                )

                if text:
                    normalized.append({
                        "text": str(text),
                        "source": item.get(
                            "source",
                            "unknown",
                        ),
                        "page": item.get(
                            "page",
                            0,
                        ),
                    })

            elif item:
                normalized.append({
                    "text": str(item),
                    "source": "unknown",
                    "page": 0,
                })

        if not normalized:
            return {
                "success": False,
                "query": query,
                "mode": mode,
                "analysis": "",
                "evidence_count": 0,
                "error": "No evidence supplied.",
            }

        # Build a grounded analysis from the supplied evidence.
        points = []

        for index, item in enumerate(
            normalized,
            start=1,
        ):
            points.append(
                f"[{index}] {item['text']}"
            )

        analysis = (
            "The retrieved evidence supports the following "
            "points relevant to the query: "
            + " ".join(points)
        )

        return {
            "success": True,
            "query": query,
            "mode": mode,
            "analysis": analysis,
            "evidence_count": len(
                normalized
            ),
            "evidence": normalized,
        }