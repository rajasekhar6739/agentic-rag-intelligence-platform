from datetime import datetime
from typing import Any


class AgentTracer:

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def log(
        self,
        event: str,
        **data,
    ):
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            **data,
        })

    def start(self, query: str):
        self.log(
            "agent_start",
            query=query,
        )

    def plan(self, plan: dict):
        self.log(
            "plan_created",
            plan=plan,
        )

    def tool(
        self,
        tool: str,
        result: Any,
    ):
        self.log(
            "tool_executed",
            tool=tool,
            result=result,
        )

    def critic(
        self,
        result: dict,
    ):
        self.log(
            "critic_evaluation",
            result=result,
        )

    def finish(
        self,
        result: dict,
    ):
        self.log(
            "agent_finish",
            grounded=result.get("grounded"),
            confidence=result.get("confidence"),
            attempts=result.get("final_attempt"),
        )

    def get_events(self):
        return list(self.events)

    def clear(self):
        self.events.clear()