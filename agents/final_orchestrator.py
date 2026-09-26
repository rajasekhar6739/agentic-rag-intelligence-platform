class FinalAgentOrchestrator:

    def __init__(
        self,
        planner,
        executor,
        synthesizer,
        critic,
        max_retries=2
    ):

        self.planner = planner
        self.executor = executor
        self.synthesizer = synthesizer
        self.critic = critic
        self.max_retries = max_retries

    def run(self, query):

        attempts = []

        for attempt in range(
            self.max_retries + 1
        ):

            plan = self.planner.plan(
                query
            )

            if not plan.get("tasks"):

                return {
                    "answer": (
                        "Unable to create "
                        "an execution plan."
                    ),
                    "grounded": False,
                    "attempts": attempts
                }

            results = self.executor.execute(
                plan
            )

            final = self.synthesizer.synthesize(
                query=query,
                results=results
            )

            evidence = []

            for item in results:

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
                query=query,
                answer=final["answer"],
                evidence=evidence
            )

            attempts.append({
                "attempt": attempt + 1,
                "plan": plan,
                "critique": critique
            })

            if critique.get(
                "grounded",
                False
            ):

                final["attempts"] = attempts

                return final

            # Force the next planning cycle
            # to focus on the critic's issues.
            issues = critique.get(
                "issues",
                []
            )

            query = (
                query
                + "\n\nPrevious attempt had "
                  "these grounding issues:\n"
                + "\n".join(
                    str(issue)
                    for issue in issues
                )
                + "\nUse only directly supported "
                  "evidence."
            )

        return {
            "answer": (
                "The system could not produce "
                "a sufficiently grounded answer."
            ),
            "grounded": False,
            "confidence": 0.0,
            "citations": [],
            "tools_used": [],
            "attempts": attempts
        }