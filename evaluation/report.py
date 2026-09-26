from evaluation.metrics import RAGMetrics
from evaluation.agent_metrics import AgentMetrics


class EvaluationReport:

    def generate(
        self,
        retrieved: list,
        relevant: list,
        execution_trace: list,
        attempts: list,
        results: list
    ) -> dict:

        return {
            "retrieval": {
                "precision_at_k": (
                    RAGMetrics.precision_at_k(
                        retrieved,
                        relevant,
                        len(retrieved)
                    )
                ),
                "recall_at_k": (
                    RAGMetrics.recall_at_k(
                        retrieved,
                        relevant,
                        len(retrieved)
                    )
                ),
                "mrr": (
                    RAGMetrics.reciprocal_rank(
                        retrieved,
                        relevant
                    )
                )
            },

            "agent": {
                "tool_success_rate": (
                    AgentMetrics.tool_success_rate(
                        execution_trace
                    )
                ),
                "retry_rate": (
                    AgentMetrics.retry_rate(
                        attempts
                    )
                ),
                "average_latency_ms": (
                    AgentMetrics.average_latency(
                        attempts
                    )
                ),
                "acceptance_rate": (
                    AgentMetrics.acceptance_rate(
                        results
                    )
                )
            }
        }