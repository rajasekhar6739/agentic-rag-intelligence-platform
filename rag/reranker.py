from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ):

        if not results:
            return []

        pairs = []

        for item in results:

            document = item.get(
                "document",
                item
            )

            if isinstance(
                document,
                dict
            ):
                text = document.get(
                    "text",
                    ""
                )
            else:
                text = str(document)

            pairs.append(
                [query, text]
            )

        scores = self.model.predict(
            pairs
        )

        ranked = []

        for item, score in zip(
            results,
            scores
        ):

            ranked.append({
                **item,
                "rerank_score": float(
                    score
                )
            })

        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked[:top_k]