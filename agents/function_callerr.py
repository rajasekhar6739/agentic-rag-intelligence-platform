import json
import os

from dotenv import load_dotenv
from groq import Groq

from agents.agent_tools import AgentTools
from agents.memory import AgentMemory
from agents.structured_output import StructuredOutput


load_dotenv()


class FunctionCaller:

    def __init__(self, retriever):

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
            "llama-3.1-8b-instant"
        )

        self.tools = AgentTools(retriever)

        self.memory = AgentMemory(
            max_turns=10
        )

    def _tool_schemas(self):

        return [
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
                                "type": "string"
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
                        "Search the knowledge base "
                        "for relevant information."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string"
                            },
                            "top_k": {
                                "type": "integer"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def run(self, user_query: str):

        self.memory.add_user_message(
            user_query
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agentic AI assistant. "
                    "Use previous conversation context "
                    "when relevant. "
                    "Use calculator for mathematics. "
                    "Use rag_search for document knowledge. "
                    "Do not invent document facts."
                )
            }
        ]

        # Add conversation memory
        for item in self.memory.get_history():

            if item["role"] in (
                "user",
                "assistant"
            ):

                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })

        executed_tools = []
        citations = []

        for _ in range(4):

            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tool_schemas(),
                    tool_choice="auto",
                    temperature=0
                )
            )

            message = response.choices[0].message

            if not message.tool_calls:

                answer = message.content or ""

                self.memory.add_assistant_message(
                    answer
                )

                return StructuredOutput.build(
                    answer=answer,
                    tool_calls=executed_tools,
                    grounded=True,
                    confidence=0.90,
                    citations=citations
                )

            messages.append(message)

            for tool_call in message.tool_calls:

                name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                result = self.tools.execute(
                    name,
                    arguments
                )

                executed_tools.append({
                    "tool": name,
                    "arguments": arguments,
                    "result": result
                })

                self.memory.add_tool_result(
                    name,
                    result
                )

                if name == "rag_search":

                    for item in result.get(
                        "results",
                        []
                    ):

                        citations.append({
                            "source": item.get(
                                "source",
                                "unknown"
                            ),
                            "page": item.get(
                                "page"
                            )
                        })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        answer = (
            "The agent reached its "
            "tool execution limit."
        )

        self.memory.add_assistant_message(
            answer
        )

        return StructuredOutput.build(
            answer=answer,
            tool_calls=executed_tools,
            grounded=False,
            confidence=0.0,
            citations=citations
        )