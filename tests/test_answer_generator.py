from rag.answer_generator import AnswerGenerator


generator = AnswerGenerator()


results = [
    {
        "document": {
            "text": (
                "Transformers use self-attention "
                "to model relationships between tokens."
            ),
            "source": "transformers.txt",
            "page": 1,
            "chunk_id": 0
        },
        "rerank_score": 4.8
    },
    {
        "document": {
            "text": (
                "Self-attention allows each token "
                "to attend to other tokens."
            ),
            "source": "attention.txt",
            "page": 2,
            "chunk_id": 1
        },
        "rerank_score": 4.2
    }
]


result = generator.generate(
    query="How does self-attention work?",
    results=results
)


print("\n==============================")
print("GROUNDED ANSWER")
print("==============================")

print(result["answer"])

print("\nSOURCES:")

for source in result["sources"]:
    print(source)