from typing import Any


class RAGController:

    def __init__(
        self,
        retriever,
        answer_generator,
        critic,
        max_retries: int = 2
    ):
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.critic = critic
        self.max_retries = max_retries

    def run(
        self,
        query: str,
        top_k: int = 5
    ) -> dict[str, Any]:

        attempts = []

        current_query = query

        for attempt in range(
            self.max_retries + 1
        ):

            results = self.retriever.search(
                current_query,
                top_k=top_k
            )

            answer_result = (
                self.answer_generator.generate(
                    query=current_query,
                    results=results
                )
            )

            answer = answer_result["answer"]

            # Critic expects answer + evidence
            evidence = []

            for item in results:

                document = item.get(
                    "document",
                    item
                )

                if isinstance(
                    document,
                    dict
                ):
                    evidence.append(
                        document.get(
                            "text",
                            ""
                        )
                    )
                else:
                    evidence.append(
                        str(document)
                    )

            critique = self.critic.evaluate(
                answer=answer,
                evidence=evidence
            )

            attempts.append({
                "attempt": attempt + 1,
                "query": current_query,
                "answer": answer,
                "critique": critique
            })

            if critique.get(
                "grounded",
                False
            ):
                return {
                    "success": True,
                    "answer": answer,
                    "sources": answer_result[
                        "sources"
                    ],
                    "attempts": attempts
                }

            # Retry with a stricter query
            current_query = (
                f"{query}\n"
                "Use only directly supported "
                "evidence. Avoid unsupported claims."
            )

        return {
            "success": False,
            "answer": (
                "I could not produce a "
                "sufficiently grounded answer."
            ),
            "sources": [],
            "attempts": attempts
        }