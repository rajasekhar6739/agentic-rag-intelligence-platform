import os
import json

from dotenv import load_dotenv
from groq import Groq

from tools.registry import ToolRegistry

load_dotenv()


class ToolAgent:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        self.registry = ToolRegistry()

    def run(
        self,
        user_request: str
    ) -> dict:

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": (
                        "Perform mathematical calculations."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": (
                                    "Mathematical expression."
                                )
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": (
                        "Search internal documents "
                        "for relevant evidence."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Question to search "
                                    "in internal documents."
                                )
                            },
                            "top_k": {
                                "type": "integer",
                                "description": (
                                    "Number of evidence "
                                    "items to retrieve."
                                )
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an enterprise AI agent. "
                    "Use tools when they are required. "
                    "Do not invent tool results."
                )
            },
            {
                "role": "user",
                "content": user_request
            }
        ]

        response = (
            self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        )

        message = response.choices[0].message

        tool_calls = message.tool_calls or []

        results = []

        for tool_call in tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = json.loads(
                tool_call.function.arguments
            )

            result = self.registry.execute(
                tool_name,
                arguments
            )

            results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

        return {
            "content": message.content,
            "tool_calls": results
        }