import json
import math

import httpx
import pytest

from tenant_memory import EmbeddingError, HashingEmbedder, OpenAIEmbedder, chunk_text


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# -- chunking -------------------------------------------------------------------------


def test_short_text_is_a_single_chunk():
    assert chunk_text("Hello there.") == ["Hello there."]


def test_empty_text_has_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_the_size_limit_and_lose_no_words():
    text = "\n\n".join(" ".join(f"w{p}x{i}" for i in range(80)) for p in range(12))
    chunks = chunk_text(text, max_chars=400, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 400 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"w{p}x{i}" in joined for p in range(12) for i in range(80))


def test_one_enormous_sentence_is_hard_split():
    chunks = chunk_text("word " * 1000, max_chars=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) >= 16


def test_chunking_is_deterministic():
    text = "Alpha beta. " * 400
    assert chunk_text(text) == chunk_text(text)


@pytest.mark.parametrize("kwargs", [{"max_chars": 50}, {"max_chars": 400, "overlap": 400}, {"max_chars": 400, "overlap": -1}])
def test_bad_chunk_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        chunk_text("x", **kwargs)


# -- hashing embedder -----------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(256)
    v1, v2 = e.embed(["teeth whitening offer"])[0], e.embed(["teeth whitening offer"])[0]
    assert v1 == v2
    assert len(v1) == 256
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)


def test_related_text_scores_higher_than_unrelated():
    e = HashingEmbedder(512)
    q, near, far = e.embed(["teeth whitening for new patients", "whitening treatment for patients", "waterfront condo closing costs"])
    assert cosine(q, near) > cosine(q, far) + 0.2


def test_empty_text_still_gets_a_usable_vector():
    v = HashingEmbedder(64).embed([""])[0]
    assert math.isclose(sum(x * x for x in v), 1.0)


# -- OpenAI embedder (no network: the transport is mocked) -----------------------------


def make(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIEmbedder("sk-test-secret", dimensions=4, client=client, **kwargs)


def test_openai_request_shape_and_ordering():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        # return out of order on purpose: the embedder must sort by index
        data = [{"index": i, "embedding": [float(i)] * 4} for i in reversed(range(len(seen["body"]["input"])))]
        return httpx.Response(200, json={"data": data})

    vectors = make(handler).embed(["a", "b", "c"])
    assert seen["url"] == "https://api.openai.com/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test-secret"
    assert seen["body"] == {"model": "text-embedding-3-large", "input": ["a", "b", "c"], "dimensions": 4}
    assert vectors == [[0.0] * 4, [1.0] * 4, [2.0] * 4]


def test_openai_batches_requests():
    calls = []

    def handler(request):
        batch = json.loads(request.content)["input"]
        calls.append(len(batch))
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [0.0] * 4} for i in range(len(batch))]})

    assert len(make(handler, batch_size=2).embed(list("abcde"))) == 5
    assert calls == [2, 2, 1]


def test_openai_failure_never_leaks_the_key():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "bad key sk-test-secret"}})

    with pytest.raises(EmbeddingError) as err:
        make(handler).embed(["a"])
    assert "sk-test-secret" not in str(err.value)
    assert "401" in str(err.value)


def test_openai_rejects_a_wrong_vector_size_and_network_errors():
    def wrong_size(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 3}]})

    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(EmbeddingError):
        make(wrong_size).embed(["a"])
    with pytest.raises(EmbeddingError):
        make(boom).embed(["a"])


def test_openai_requires_a_key():
    with pytest.raises(EmbeddingError):
        OpenAIEmbedder("")
