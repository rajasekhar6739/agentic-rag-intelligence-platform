import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class AnswerGenerator:

    def __init__(self):

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

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

    def generate(
        self,
        query: str,
        results: list
    ):

        if not results:

            return {
                "answer": (
                    "I could not find "
                    "supporting information."
                ),
                "sources": []
            }

        context_parts = []
        sources = []

        for index, item in enumerate(
            results,
            start=1
        ):

            document = item.get(
                "document",
                item
            )

            if isinstance(
                document,
                dict
            ):

                text = document.get(
                    "text",
                    ""
                )

                source = document.get(
                    "source",
                    "unknown"
                )

                page = document.get(
                    "page",
                    None
                )

                context_parts.append(
                    f"[{index}] {text}"
                )

                sources.append({
                    "id": index,
                    "source": source,
                    "page": page
                })

        context = "\n\n".join(
            context_parts
        )

        system_prompt = """
You are a grounded RAG assistant.

Answer ONLY using the supplied context.

Rules:
1. Do not invent facts.
2. If the context does not support
   the answer, say so.
3. Cite supporting context using
   [1], [2], etc.
4. Keep the answer concise.
"""

        user_prompt = f"""
CONTEXT:

{context}


QUESTION:

{query}
"""

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return {
            "answer": answer,
            "sources": sources
        }