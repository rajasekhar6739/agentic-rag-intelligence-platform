from typing import List


class RAGMetrics:

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:

        if k <= 0:
            return 0.0

        top_k = retrieved[:k]

        if not top_k:
            return 0.0

        relevant_set = set(relevant)

        hits = sum(
            1
            for item in top_k
            if item in relevant_set
        )

        return hits / len(top_k)

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:

        if not relevant:
            return 0.0

        top_k = retrieved[:k]

        relevant_set = set(relevant)

        hits = sum(
            1
            for item in top_k
            if item in relevant_set
        )

        return hits / len(relevant_set)

    @staticmethod
    def reciprocal_rank(
        retrieved: List[str],
        relevant: List[str]
    ) -> float:

        relevant_set = set(relevant)

        for index, item in enumerate(
            retrieved,
            start=1
        ):

            if item in relevant_set:
                return 1.0 / index

        return 0.0