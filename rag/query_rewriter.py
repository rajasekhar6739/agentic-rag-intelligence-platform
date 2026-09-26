import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class QueryRewriter:

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
    def rewrite(self, query: str) -> dict:

        system_prompt = """
You are a query optimization component
for an enterprise RAG system.

Given a user's question, produce
retrieval-friendly queries.

Return ONLY valid JSON:

{
  "original_query": "...",
  "rewritten_query": "...",
  "keywords": ["...", "..."],
  "sub_queries": ["...", "..."]
}

Rules:
- Preserve the user's original intent.
- Do not invent facts.
- Extract important technical terms.
- Make the rewritten query specific
  for document retrieval.
- Use sub_queries only when the question
  contains multiple information needs.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )

        content = response.choices[0].message.content

        return json.loads(content)