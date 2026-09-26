from evaluation.metrics import RAGMetrics


retrieved = [
    "doc_3",
    "doc_1",
    "doc_7",
    "doc_2"
]

relevant = [
    "doc_1",
    "doc_2"
]


precision = RAGMetrics.precision_at_k(
    retrieved,
    relevant,
    4
)

recall = RAGMetrics.recall_at_k(
    retrieved,
    relevant,
    4
)

mrr = RAGMetrics.reciprocal_rank(
    retrieved,
    relevant
)


print("\n==============================")
print("RAG EVALUATION")
print("==============================")

print(
    "Precision@4:",
    precision
)

print(
    "Recall@4:",
    recall
)

print(
    "MRR:",
    mrr
)