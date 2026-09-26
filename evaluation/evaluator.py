class AgentEvaluator:

    def evaluate(self, result):

        if not isinstance(result, dict):
            return {
                "passed": False,
                "score": 0.0,
                "issues": ["Invalid result."]
            }

        answer = result.get(
            "answer",
            ""
        )

        grounded = bool(
            result.get(
                "grounded",
                False
            )
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        evidence = result.get(
            "evidence",
            []
        )

        citations = result.get(
            "citations",
            []
        )

        checks = {
            "answer_present": bool(
                answer.strip()
            ),
            "grounded": grounded,
            "evidence_found": bool(
                evidence
            ),
            "citations_present": bool(
                citations
            ),
            "confidence_valid": (
                0.0 <= confidence <= 1.0
            )
        }

        passed_checks = sum(
            checks.values()
        )

        score = (
            passed_checks
            / len(checks)
        )

        return {
            "passed": all(
                checks.values()
            ),
            "score": score,
            "checks": checks,
            "confidence": confidence,
            "evidence_count": len(evidence),
            "citation_count": len(citations)
        }