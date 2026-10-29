"""Tests for the Gemini embedder's own logic — no network, no key.

The model is Google's; what is ours is the contract around it: vectors come out
unit-length whatever size was asked for, passages and questions are embedded as
different task types, large inputs are batched, and a free-tier 429 means wait
and retry rather than lose a half-built index.
"""

import math

import pytest
from google.genai import errors

import vectorstore


class _Embedding:
    def __init__(self, values):
        self.values = values


class _Models:
    def __init__(self, fail_first=0, code=429):
        self.calls, self.fail_first, self.code = [], fail_first, code

    def embed_content(self, model, contents, config):
        self.calls.append((len(contents), config.task_type, config.output_dimensionality))
        if len(self.calls) <= self.fail_first:
            raise errors.APIError(self.code, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        # Deliberately not unit-length, as truncated Gemini vectors are not.
        return type("Response", (), {"embeddings": [_Embedding([3.0, 4.0]) for _ in contents]})


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(vectorstore.time, "sleep", lambda _: None)
    instance = vectorstore.GeminiEmbeddings(dim=768, tokens_per_minute=10**9)
    instance._client = type("Client", (), {"models": _Models()})()
    return instance


def test_vectors_are_normalised(embedder):
    (vector,) = embedder.embed_documents(["clause"])
    assert math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-6)


def test_passages_and_questions_use_different_task_types(embedder):
    embedder.embed_documents(["clause"])
    embedder.embed_query("question?")
    tasks = [task for _, task, _ in embedder._client.models.calls]
    assert tasks == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_requested_size_is_sent(embedder):
    embedder.embed_query("question?")
    assert embedder._client.models.calls[0][2] == 768


def test_documents_are_batched(embedder, monkeypatch):
    monkeypatch.setattr(vectorstore, "EMBED_BATCH", 4)
    assert len(embedder.embed_documents(["x"] * 10)) == 10
    assert [n for n, _, _ in embedder._client.models.calls] == [4, 4, 2]


def test_rate_limit_is_retried(embedder):
    embedder._client.models.fail_first = 2
    assert len(embedder.embed_documents(["clause"])) == 1
    assert len(embedder._client.models.calls) == 3


def test_a_bad_request_is_not_retried(embedder):
    embedder._client.models.fail_first, embedder._client.models.code = 1, 400
    with pytest.raises(errors.APIError):
        embedder.embed_query("question?")
    assert len(embedder._client.models.calls) == 1


def test_missing_key_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="aistudio.google.com"):
        vectorstore.GeminiEmbeddings()
