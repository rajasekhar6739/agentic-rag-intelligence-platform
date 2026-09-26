import ast
import json
import operator
import os

from dotenv import load_dotenv
from groq import Groq

from agents.agent_tools import AgentTools
from agents.memory import AgentMemory
from agents.structured_output import StructuredOutput


load_dotenv()


class FunctionCaller:

    def __init__(self, retriever=None):
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
            "llama-3.1-8b-instant",
        )

        self.retriever = retriever

        self.tools = AgentTools(
            retriever
        )

        self.memory = AgentMemory(
            max_turns=10
        )

    # ============================================================
    # SAFE CALCULATOR
    # ============================================================

    def _calculate(self, expression):
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def evaluate(node):
            if isinstance(node, ast.Constant):
                if isinstance(
                    node.value,
                    (int, float),
                ):
                    return node.value

                raise ValueError(
                    "Invalid numeric constant."
                )

            if isinstance(node, ast.constant):
                return node.n

            if isinstance(node, ast.UnaryOp):
                operation = operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Unsupported unary operator."
                    )

                return operation(
                    evaluate(node.operand)
                )

            if isinstance(node, ast.BinOp):
                operation = operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Unsupported operator."
                    )

                return operation(
                    evaluate(node.left),
                    evaluate(node.right),
                )

            raise ValueError(
                "Invalid mathematical expression."
            )

        tree = ast.parse(
            expression,
            mode="eval",
        )

        return evaluate(
            tree.body
        )

    # ============================================================
    # TOOL SCHEMAS
    # ============================================================

    def _tool_schemas(self):

        schemas = [
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
                        "required": [
                            "expression"
                        ],
                    },
                },
            }
        ]

        if self.retriever is not None:
            schemas.append(
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
                                },
                            },
                            "required": [
                                "query"
                            ],
                        },
                    },
                }
            )

        return schemas

    # ============================================================
    # LOCAL TOOL EXECUTION
    # ============================================================

    def _execute_tool(
        self,
        name,
        arguments,
    ):

        if name == "calculator":
            expression = arguments.get(
                "expression",
                "",
            )

            try:
                value = self._calculate(
                    expression
                )

                return {
                    "success": True,
                    "tool": "calculator",
                    "result": value,
                }

            except Exception as exc:
                return {
                    "success": False,
                    "tool": "calculator",
                    "error": str(exc),
                }

        if name == "rag_search":

            if self.retriever is None:
                return {
                    "success": False,
                    "tool": "rag_search",
                    "error": (
                        "RAG retriever is not configured."
                    ),
                }

        try:
            return self.tools.execute(
                name,
                arguments,
            )

        except Exception as exc:
            return {
                "success": False,
                "tool": name,
                "error": str(exc),
            }

    # ============================================================
    # RUN
    # ============================================================

    def run(self, user_query: str):

        self.memory.add_user_message(
            user_query
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agentic AI assistant. "
                    "Use conversation context when relevant. "
                    "Use calculator for mathematics. "
                    "Use rag_search for document knowledge "
                    "when available. "
                    "Never invent document facts."
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        executed_tools = []
        citations = []

        for _ in range(4):

            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tool_schemas(),
                    tool_choice="auto",
                    temperature=0,
                )
            )

            message = response.choices[
                0
            ].message

            if not message.tool_calls:

                answer = (
                    message.content
                    or ""
                )

                self.memory.add_assistant_message(
                    answer
                )

                return StructuredOutput.build(
                    answer=answer,
                    tool_calls=executed_tools,
                    grounded=bool(
                        citations
                    )
                    or not self.retriever,
                    confidence=(
                        0.90
                        if not citations
                        else 0.90
                    ),
                    citations=citations,
                )

            messages.append(
                message
            )

            for tool_call in message.tool_calls:

                name = (
                    tool_call.function.name
                )

                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )
                except Exception:
                    arguments = {}

                result = self._execute_tool(
                    name,
                    arguments,
                )

                executed_tools.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )

                self.memory.add_tool_result(
                    name,
                    result,
                )

                if name == "rag_search":

                    for item in result.get(
                        "results",
                        [],
                    ):

                        citations.append(
                            {
                                "source": item.get(
                                    "source",
                                    "unknown",
                                ),
                                "page": item.get(
                                    "page"
                                ),
                            }
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": json.dumps(
                            result
                        ),
                    }
                )

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
            citations=citations,
        )