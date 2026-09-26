from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


class AgentMetrics:
    """
    Metrics and evaluation utilities for the agentic RAG system.
    """

    @staticmethod
    def _items(results: Any) -> list[Mapping[str, Any]]:
        if results is None:
            return []

        if isinstance(results, Mapping):
            return [results]

        if isinstance(results, Iterable) and not isinstance(
            results, (str, bytes)
        ):
            return [
                item for item in results
                if isinstance(item, Mapping)
            ]

        return []

    @staticmethod
    def _tool_calls(
        result: Mapping[str, Any]
    ) -> list[Mapping[str, Any]]:
        calls = result.get("tool_calls", [])

        if isinstance(calls, list):
            return [
                call for call in calls
                if isinstance(call, Mapping)
            ]

        return []

    @classmethod
    def grounded_rate(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        grounded = sum(
            1
            for item in items
            if item.get("grounded") is True
        )

        return round(grounded / len(items), 4)

    @classmethod
    def success_rate(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        successful = sum(
            1
            for item in items
            if item.get("success") is True
        )

        return round(successful / len(items), 4)

    @classmethod
    def tool_success_rate(cls, results: Any) -> float:
        items = cls._items(results)

        calls: list[Mapping[str, Any]] = []

        for item in items:
            calls.extend(cls._tool_calls(item))

        if not calls:
            return 0.0

        successful = 0

        for call in calls:
            result = call.get("result")

            if isinstance(result, Mapping):
                if result.get("success") is True:
                    successful += 1
            elif call.get("success") is True:
                successful += 1

        return round(successful / len(calls), 4)

    @classmethod
    def retry_rate(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        retries = 0

        for item in items:
            attempt_count = item.get("attempt_count")

            if isinstance(attempt_count, (int, float)):
                if attempt_count > 1:
                    retries += 1
                    continue

            attempts = item.get("attempts")

            if isinstance(attempts, list) and len(attempts) > 1:
                retries += 1
                continue

            if item.get("self_corrected") is True:
                retries += 1

        return round(retries / len(items), 4)

    @classmethod
    def average_latency(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        values: list[float] = []

        keys = (
            "latency_ms",
            "latency",
            "response_time_ms",
            "response_time",
            "duration_ms",
            "duration",
        )

        for item in items:
            value: Optional[Any] = None

            for key in keys:
                if key in item:
                    value = item.get(key)
                    break

            if isinstance(value, (int, float)):
                values.append(float(value))

        if not values:
            return 0.0

        return round(sum(values) / len(values), 4)

    @classmethod
    def average_confidence(cls, results: Any) -> float:
        items = cls._items(results)

        values = [
            float(item["confidence"])
            for item in items
            if isinstance(item.get("confidence"), (int, float))
        ]

        if not values:
            return 0.0

        return round(sum(values) / len(values), 4)

    @classmethod
    def average_evidence(cls, results: Any) -> float:
        items = cls._items(results)

        values: list[float] = []

        for item in items:
            if isinstance(item.get("evidence_count"), (int, float)):
                values.append(float(item["evidence_count"]))
                continue

            evidence = item.get("evidence")

            if isinstance(evidence, list):
                values.append(float(len(evidence)))

        if not values:
            return 0.0

        return round(sum(values) / len(values), 4)

    @classmethod
    def total_tool_calls(cls, results: Any) -> int:
        return sum(
            len(cls._tool_calls(item))
            for item in cls._items(results)
        )

    @classmethod
    def average_tools_per_run(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        return round(
            cls.total_tool_calls(items) / len(items),
            4,
        )

    @classmethod
    def citation_rate(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        cited = 0

        for item in items:
            citations = item.get("citations", [])

            if isinstance(citations, list) and citations:
                cited += 1

        return round(cited / len(items), 4)

    @classmethod
    def acceptance_rate(cls, results: Any) -> float:
        items = cls._items(results)

        if not items:
            return 0.0

        accepted = 0

        for item in items:
            if item.get("accepted") is True:
                accepted += 1
            elif item.get("success") is True:
                accepted += 1

        return round(accepted / len(items), 4)

    def evaluate(self, results: Any) -> dict[str, Any]:
        items = self._items(results)

        if not items:
            return {
                "total": 0,
                "successful": 0,
                "success_rate": 0.0,
                "grounded": 0,
                "grounded_rate": 0.0,
                "average_confidence": 0.0,
                "average_evidence": 0.0,
                "tool_usage_rate": 0.0,
                "tool_success_rate": 0.0,
                "retry_rate": 0.0,
                "average_latency": 0.0,
                "acceptance_rate": 0.0,
            }

        total = len(items)

        successful = sum(
            1
            for item in items
            if item.get("success") is True
        )

        grounded = sum(
            1
            for item in items
            if item.get("grounded") is True
        )

        tool_usage = sum(
            1
            for item in items
            if item.get("tools_used")
        )

        return {
            "total": total,
            "successful": successful,
            "success_rate": round(successful / total, 4),
            "grounded": grounded,
            "grounded_rate": round(grounded / total, 4),
            "average_confidence": self.average_confidence(items),
            "average_evidence": self.average_evidence(items),
            "tool_usage_rate": round(tool_usage / total, 4),
            "tool_success_rate": self.tool_success_rate(items),
            "retry_rate": self.retry_rate(items),
            "average_latency": self.average_latency(items),
            "acceptance_rate": self.acceptance_rate(items),
        }

    @classmethod
    def summary(cls, results: Any) -> dict[str, Any]:
        items = cls._items(results)

        return {
            "runs": len(items),
            "success_rate": cls.success_rate(items),
            "grounded_rate": cls.grounded_rate(items),
            "tool_success_rate": cls.tool_success_rate(items),
            "retry_rate": cls.retry_rate(items),
            "average_latency": cls.average_latency(items),
            "average_confidence": cls.average_confidence(items),
            "average_evidence": cls.average_evidence(items),
            "citation_rate": cls.citation_rate(items),
            "acceptance_rate": cls.acceptance_rate(items),
            "total_tool_calls": cls.total_tool_calls(items),
            "average_tools_per_run": cls.average_tools_per_run(items),
        }