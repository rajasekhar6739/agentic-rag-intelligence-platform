from rag.query_rewriter import QueryRewriter
from rag.retriever import HybridRetriever
from rag.reranker import Reranker
from agents.schemas import DocumentChunk


class RAGPipeline:

    def __init__(self):

        self.query_rewriter = QueryRewriter()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

    def build(
        self,
        documents: list[DocumentChunk]
    ):

        self.retriever.build(documents)

    def search(
        self,
        query: str,
        candidate_k: int = 10,
        top_k: int = 5
    ) -> dict:

        # 1. Query understanding
        rewritten = self.query_rewriter.rewrite(
            query
        )

        retrieval_query = (
            rewritten["rewritten_query"]
        )

        # 2. Hybrid retrieval
        candidates = self.retriever.search(
            retrieval_query,
            top_k=candidate_k
        )

        # 3. Cross-encoder reranking
        evidence = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

        return {
            "original_query": query,
            "rewritten_query": retrieval_query,
            "keywords": rewritten["keywords"],
            "sub_queries": rewritten["sub_queries"],
            "evidence": evidence
        }