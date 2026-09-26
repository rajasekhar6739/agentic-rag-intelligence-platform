from multimodal.document_adapter import (
    MultimodalDocumentAdapter
)


description = """
Production dashboard image.

API latency: 850 ms.
CPU utilization: 92%.
Database connection pool:
100/100 connections used.
"""


chunk = MultimodalDocumentAdapter.to_chunk(
    description=description,
    source="dashboard.png",
    page=1,
    chunk_id=0
)


print("\n==============================")
print("MULTIMODAL DOCUMENT")
print("==============================")

print(chunk)