from llm.client import LLMClient


def test_llm_client_fallback():

    client = LLMClient(
        api_key=None
    )

    result = client.generate(
        "Explain RAG in one sentence."
    )

    assert result["success"] is True
    assert isinstance(
        result["text"],
        str,
    )
    assert result["text"]