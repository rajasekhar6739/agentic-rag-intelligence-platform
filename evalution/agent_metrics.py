class AgentMetrics:

    @staticmethod
    def tool_success_rate(
        execution_trace: list
    ) -> float:

        if not execution_trace:
            return 0.0

        successful = 0

        for event in execution_trace:

            result = event.get(
                "result",
                {}
            )

            if result.get("success") is True:
                successful += 1

        return successful / len(
            execution_trace
        )

    @staticmethod
    def retry_rate(
        attempts: list
    ) -> float:

        if not attempts:
            return 0.0

        retries = max(
            len(attempts) - 1,
            0
        )

        return retries / len(
            attempts
        )

    @staticmethod
    def average_latency(
        attempts: list
    ) -> float:

        if not attempts:
            return 0.0

        latencies = [
            attempt.get(
                "latency_ms",
                0
            )
            for attempt in attempts
        ]

        return sum(latencies) / len(
            latencies
        )

    @staticmethod
    def acceptance_rate(
        results: list
    ) -> float:

        if not results:
            return 0.0

        accepted = sum(
            1
            for result in results
            if result.get("accepted") is True
        )

        return accepted / len(
            results
        )