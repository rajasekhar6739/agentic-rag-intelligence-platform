import os
from dotenv import load_dotenv

from rag.query_rewriter import QueryRewriter


load_dotenv()


rewriter = QueryRewriter()


query = """
Why did API latency increase during yesterday's
incident and what evidence explains the root cause?
"""


result = rewriter.rewrite(query)


print("\nORIGINAL:")
print(result["original_query"])

print("\nREWRITTEN:")
print(result["rewritten_query"])

print("\nKEYWORDS:")
for keyword in result["keywords"]:
    print("-", keyword)

print("\nSUB QUERIES:")
for sub_query in result["sub_queries"]:
    print("-", sub_query)