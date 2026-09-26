from rag.pipeline import RAGPipeline


class RAGSearchTool:

    name = "rag_search"

    description = (
        "Search the knowledge base and return "
        "relevant evidence for a user query."
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(
        self,
        query: str,
        top_k: int = 5
    ):

        if not query or not query.strip():

            return {
                "success": False,
                "query": query,
                "results": [],
                "error": "Query cannot be empty."
            }

        try:

            results = self.retriever.search(
                query,
                top_k=top_k
            )

            return {
                "success": True,
                "query": query,
                "results": results
            }

        except Exception as exc:

            return {
                "success": False,
                "query": query,
                "results": [],
                "error": str(exc)
            }