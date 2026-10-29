"""Tests for the Cohere embedder's own logic — no network, no key.

The model is Cohere's; what is ours is the contract around it: vectors come out
unit-length, passages and questions are embedded as different input types,
large inputs are batched at Cohere's 96-text maximum, and a rate limit means
wait and retry rather than lose a half-built index.
"""

import math

import cohere
import pytest

import vectorstore


class _Client:
    def __init__(self, fail_first=0, error=None):
        self.calls, self.fail_first, self.error = [], fail_first, error

    def embed(self, model, input_type, texts, embedding_types, output_dimension):
        self.calls.append((len(texts), input_type, output_dimension))
        if len(self.calls) <= self.fail_first:
            raise self.error
        # Deliberately not unit-length.
        embeddings = type("Embeddings", (), {"float_": [[3.0, 4.0] for _ in texts]})
        return type("Response", (), {"embeddings": embeddings})


def _rate_limited():
    return cohere.TooManyRequestsError(body={"message": "trial limit"})


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test")
    monkeypatch.setattr(vectorstore.time, "sleep", lambda _: None)
    instance = vectorstore.CohereEmbeddings(dim=1024)
    instance._client = _Client()
    return instance


def test_vectors_are_normalised(embedder):
    (vector,) = embedder.embed_documents(["clause"])
    assert math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-6)


def test_passages_and_questions_use_different_input_types(embedder):
    embedder.embed_documents(["clause"])
    embedder.embed_query("question?")
    assert [kind for _, kind, _ in embedder._client.calls] == ["search_document", "search_query"]


def test_requested_size_is_sent(embedder):
    embedder.embed_query("question?")
    assert embedder._client.calls[0][2] == 1024


def test_documents_are_batched(embedder, monkeypatch):
    monkeypatch.setattr(vectorstore, "EMBED_BATCH", 4)
    assert len(embedder.embed_documents(["x"] * 10)) == 10
    assert [n for n, _, _ in embedder._client.calls] == [4, 4, 2]


def test_rate_limit_is_retried(embedder):
    embedder._client = _Client(fail_first=2, error=_rate_limited())
    assert len(embedder.embed_documents(["clause"])) == 1
    assert len(embedder._client.calls) == 3


def test_a_bad_request_is_not_retried(embedder):
    embedder._client = _Client(fail_first=1, error=cohere.BadRequestError(body={"message": "bad"}))
    with pytest.raises(cohere.BadRequestError):
        embedder.embed_query("question?")
    assert len(embedder._client.calls) == 1


def test_missing_key_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="dashboard.cohere.com"):
        vectorstore.CohereEmbeddings()


def test_calls_are_paced_to_the_token_budget(embedder, monkeypatch):
    # Two 60-token calls against a 100-token minute: the second must wait.
    waits = []
    monkeypatch.setattr(vectorstore.time, "sleep", waits.append)
    clock = iter([0.0, 1.0, 61.0, 61.0])
    monkeypatch.setattr(vectorstore.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(vectorstore, "token_length", lambda text: 60)
    embedder.tpm = 100
    embedder.embed_query("a")
    embedder.embed_query("b")
    assert waits and waits[0] > 50
