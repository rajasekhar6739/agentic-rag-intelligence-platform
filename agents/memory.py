from __future__ import annotations

from typing import Any


class AgentMemory:

    def __init__(
        self,
        max_items: int = 50,
        max_turns: int | None = None,
        **kwargs,
    ):
        if max_turns is not None:
            max_items = max_turns

        self.max_items = max(1, int(max_items))
        self.max_turns = self.max_items

        self._items: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []
        self._tool_results: list[dict[str, Any]] = []

    # ============================================================
    # TURN MEMORY
    # ============================================================

    def add(
        self,
        query: str = "",
        answer: str = "",
        metadata: dict | None = None,
        **kwargs,
    ):
        item = {
            "query": str(query),
            "answer": str(answer),
            "metadata": metadata or {},
        }

        self._items.append(item)

        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items:]

        return item

    def add_turn(
        self,
        query: str = "",
        answer: str = "",
        metadata: dict | None = None,
        **kwargs,
    ):
        return self.add(
            query=query,
            answer=answer,
            metadata=metadata,
            **kwargs,
        )

    def remember(
        self,
        query: str = "",
        answer: str = "",
        metadata: dict | None = None,
        **kwargs,
    ):
        return self.add(
            query=query,
            answer=answer,
            metadata=metadata,
            **kwargs,
        )

    # ============================================================
    # CHAT MESSAGES
    # ============================================================

    def add_message(
        self,
        role: str,
        content: str,
        **kwargs,
    ):
        message = {
            "role": str(role),
            "content": str(content),
        }

        message.update(kwargs)

        self._messages.append(message)
        self._trim_messages()

        return message

    def add_user_message(self, content: str):
        return self.add_message(
            "user",
            content,
        )

    def add_assistant_message(self, content: str):
        return self.add_message(
            "assistant",
            content,
        )

    def add_system_message(self, content: str):
        return self.add_message(
            "system",
            content,
        )

    # ============================================================
    # TOOL RESULTS
    # ============================================================

    def add_tool_result(
        self,
        tool_name: str = "",
        result: Any = None,
        **kwargs,
    ):
        """
        Store a tool execution result.

        Compatible with FunctionCaller and tool-agent
        implementations.
        """

        item = {
            "tool": str(tool_name),
            "result": result,
        }

        item.update(kwargs)

        self._tool_results.append(item)

        if len(self._tool_results) > self.max_items:
            self._tool_results = (
                self._tool_results[-self.max_items:]
            )

        # Also expose the tool result in chat history.
        self._messages.append(
            {
                "role": "tool",
                "name": str(tool_name),
                "content": str(result),
            }
        )

        self._trim_messages()

        return item

    def get_tool_results(
        self,
        limit: int | None = None,
    ):
        if limit is None:
            return list(self._tool_results)

        limit = max(0, int(limit))

        if limit == 0:
            return []

        return list(
            self._tool_results[-limit:]
        )

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
        limit: int | None = None,
    ):
        if limit is None:
            return list(self._messages)

        limit = max(0, int(limit))

        if limit == 0:
            return []

        return list(
            self._messages[-limit:]
        )

    def get_messages(
        self,
        limit: int | None = None,
    ):
        return self.get_history(limit)

    def messages(
        self,
        limit: int | None = None,
    ):
        return self.get_history(limit)

    def history(
        self,
        limit: int | None = None,
    ):
        return self.get_history(limit)

    # ============================================================
    # INTERNAL
    # ============================================================

    def _trim_messages(self):
        max_messages = self.max_items * 3

        if len(self._messages) > max_messages:
            self._messages = (
                self._messages[-max_messages:]
            )

    # ============================================================
    # RECENT TURNS
    # ============================================================

    def get_recent(self, limit: int = 5):
        limit = max(0, int(limit))

        if limit == 0:
            return []

        return list(
            self._items[-limit:]
        )

    def recent(self, limit: int = 5):
        return self.get_recent(limit)

    def get_all(self):
        return list(self._items)

    def all(self):
        return self.get_all()

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ):
        limit = max(0, int(limit))

        if limit == 0:
            return []

        query_tokens = set(
            str(query).lower().split()
        )

        if not query_tokens:
            return self.get_recent(limit)

        scored = []

        for item in self._items:
            text = (
                f"{item.get('query', '')} "
                f"{item.get('answer', '')}"
            ).lower()

            tokens = set(text.split())

            score = len(
                query_tokens & tokens
            )

            if score > 0:
                scored.append(
                    (score, item)
                )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            item
            for _, item in scored[:limit]
        ]

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self):
        self._items.clear()
        self._messages.clear()
        self._tool_results.clear()

    reset = clear

    def clear_messages(self):
        self._messages.clear()

    def clear_turns(self):
        self._items.clear()

    # ============================================================
    # METADATA
    # ============================================================

    def __len__(self):
        return len(self._items)

    def size(self):
        return len(self._items)

    def message_count(self):
        return len(self._messages)

    def is_empty(self):
        return (
            not self._items
            and not self._messages
            and not self._tool_results
        )

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def to_dict(self):
        return {
            "turns": self.get_all(),
            "messages": self.get_history(),
            "tool_results": self.get_tool_results(),
            "size": len(self._items),
            "message_count": len(self._messages),
        }

    def __repr__(self):
        return (
            f"AgentMemory("
            f"turns={len(self._items)}, "
            f"messages={len(self._messages)}, "
            f"tools={len(self._tool_results)})"
        )