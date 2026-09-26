class SelfCorrector:

    def __init__(
        self,
        agent,
        critic,
        max_retries=2
    ):
        self.agent = agent
        self.critic = critic
        self.max_retries = max_retries

    def run(self, query):

        attempts = []

        current_query = query

        for attempt in range(
            self.max_retries + 1
        ):

            result = self.agent.run(
                current_query
            )

            answer = result.get(
                "answer",
                ""
            )

            evidence = []

            for item in result.get(
                "results",
                []
            ):

                tool_result = item.get(
                    "result",
                    {}
                )

                if not isinstance(
                    tool_result,
                    dict
                ):
                    continue

                for document in tool_result.get(
                    "results",
                    []
                ):

                    if isinstance(
                        document,
                        dict
                    ):
                        text = document.get(
                            "text",
                            ""
                        )

                        if text:
                            evidence.append(
                                text
                            )

            critique = self.critic.evaluate(
                query=current_query,
                answer=answer,
                evidence=evidence
            )

            attempts.append({
                "attempt": attempt + 1,
                "answer": answer,
                "critique": critique
            })

            if critique.get(
                "grounded",
                False
            ):
                result["attempts"] = attempts
                return result

            current_query = (
                query
                + "\n\n"
                + "Previous answer was not "
                  "sufficiently grounded. "
                  "Use only directly supported "
                  "evidence and avoid assumptions."
            )

        return {
            "answer": (
                "I could not produce a "
                "sufficiently grounded answer."
            ),
            "grounded": False,
            "confidence": 0.0,
            "citations": [],
            "tools_used": [],
            "attempts": attempts
        }