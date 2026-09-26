import json

from agents.schemas import AgentAnswer


class ResponseParser:

    @staticmethod
    def parse(
        content: str
    ) -> AgentAnswer:

        try:

            data = json.loads(content)

            return AgentAnswer.model_validate(
                data
            )

        except Exception as exc:

            raise ValueError(
                "LLM response does not match "
                "AgentAnswer schema."
            ) from exc