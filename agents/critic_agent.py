from agents.critic import Critic


class CriticAgent:

    def __init__(self, retriever=None):
        self.retriever = retriever
        self.critic = Critic()

    def evaluate(
        self,
        query,
        answer,
        evidence
    ):
        return self.critic.evaluate(
            query=query,
            answer=answer,
            evidence=evidence
        )