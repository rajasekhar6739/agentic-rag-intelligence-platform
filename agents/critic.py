
import re


class Critic:

    INSUFFICIENT_PATTERNS = [
        "evidence is insufficient",
        "provided evidence is insufficient",
        "not enough information",
        "does not contain information",
        "doesn't contain information",
        "no information",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not provided in the evidence",
        "not found in the evidence",
        "insufficient information",
    ]

    STOP_WORDS = {
        "about", "after", "again", "also", "answer",
        "because", "being", "between", "could", "does",
        "from", "have", "into", "more", "only", "other",
        "should", "some", "such", "than", "that", "their",
        "there", "these", "they", "this", "those", "through",
        "using", "which", "with", "would", "your",
        "what", "when", "where", "which", "were",
        "while", "then", "than", "been", "being",
    }

    def evaluate(self, *args, **kwargs):
        """
        Supports:

            evaluate(answer, evidence)

        and:

            evaluate(query, answer, evidence)
        """

        query = kwargs.get("query")
        answer = kwargs.get("answer")
        evidence = kwargs.get("evidence")

        if len(args) == 2:
            answer, evidence = args

        elif len(args) == 3:
            query, answer, evidence = args

        elif len(args) == 0:
            pass

        else:
            raise TypeError(
                "evaluate() expects "
                "(answer, evidence) or "
                "(query, answer, evidence)"
            )

        if answer is None:
            answer = ""

        if evidence is None:
            evidence = []

        issues = []
        supported = True

        if not str(answer).strip():
            return {
                "query": query,
                "supported": False,
                "issues": ["Answer is empty."],
                "confidence": 0.0,
            }

        if not evidence:
            return {
                "query": query,
                "supported": False,
                "issues": ["No evidence was retrieved."],
                "confidence": 0.0,
            }

        answer_text = str(answer).strip()
        answer_lower = answer_text.lower()

        insufficient = any(
            pattern in answer_lower
            for pattern in self.INSUFFICIENT_PATTERNS
        )

        if insufficient:
            return {
                "query": query,
                "supported": False,
                "issues": [
                    "The answer explicitly states that "
                    "the retrieved evidence is insufficient."
                ],
                "confidence": 0.0,
            }

        evidence_parts = []

        for item in evidence:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")

            if text:
                evidence_parts.append(str(text))

        evidence_text = " ".join(
            evidence_parts
        ).lower()

        if not evidence_text.strip():
            return {
                "query": query,
                "supported": False,
                "issues": [
                    "Retrieved evidence contains no usable text."
                ],
                "confidence": 0.0,
            }

        answer_without_citations = re.sub(
            r"\[\d+\]",
            "",
            answer_text,
        )

        answer_words = re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
            answer_without_citations.lower(),
        )

        meaningful_words = [
            word
            for word in answer_words
            if len(word) >= 4
            and word not in self.STOP_WORDS
        ]

        matched_words = [
            word
            for word in meaningful_words
            if word in evidence_text
        ]

        unique_meaningful = set(
            meaningful_words
        )

        unique_matched = set(
            matched_words
        )

        overlap_ratio = (
            len(unique_matched)
            / max(
                len(unique_meaningful),
                1,
            )
        )

        citations = re.findall(
            r"\[(\d+)\]",
            answer_text,
        )

        invalid_citations = []

        for citation in citations:
            index = int(citation)

            if index < 1 or index > len(evidence):
                invalid_citations.append(index)

        if invalid_citations:
            issues.append(
                "Answer contains invalid evidence citations: "
                + str(invalid_citations)
            )
            supported = False

        if overlap_ratio < 0.20:
            issues.append(
                "Answer does not have sufficient lexical overlap "
                "with the retrieved evidence."
            )
            supported = False

        if supported:
            confidence = min(
                1.0,
                0.70 + overlap_ratio * 0.30,
            )
        else:
            confidence = 0.0

        return {
            "query": query,
            "supported": supported,
            "issues": issues,
            "confidence": confidence,
            "overlap_ratio": round(
                overlap_ratio,
                4,
            ),
            "matched_words": sorted(
                unique_matched
            ),
        }