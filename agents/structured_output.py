import json


class StructuredOutput:
    """
    Standard output format for AURA agents.

    Keeps both:
        - tool_calls: complete execution history
        - tools_used: unique tool names
    """

    @staticmethod
    def build(
        answer,
        tool_calls=None,
        grounded=True,
        confidence=0.0,
        citations=None,
    ):
        tool_calls = tool_calls or []
        citations = citations or []

        tools_used = []

        for call in tool_calls:
            if not isinstance(call, dict):
                continue

            name = call.get("tool")

            if name and name not in tools_used:
                tools_used.append(name)

        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "grounded": bool(grounded),
            "confidence": float(confidence),
            "citations": citations,
            "tools_used": tools_used,
        }

    @staticmethod
    def to_json(data):
        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )