"""Chroma vector store for the compliance corpus.

The single source of truth for how documents are embedded and where they live.
Ingestion and retrieval both import from here on purpose: if the two sides ever
built the embedder differently the vectors would land in a different space than
the queries, and retrieval would quietly return noise rather than fail.

No Streamlit or chat imports, matching store.py, so this stays usable from the
CLI and from a future API layer.
"""

import logging
import os
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

# chromadb 0.5.23 calls posthog.capture() with an argument list posthog 7.x no
# longer accepts, so it logs a failure per client even with telemetry off below.
# Nothing is sent either way; this just keeps the noise out of the CLI output.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Gemini embedding model and output size. Both are stamped onto the collection
# when it is created and checked on every open: vectors from two models (or two
# sizes of one) live in different spaces, and querying one with the other
# returns noise rather than an error. gemini-embedding-001 is trained so that
# 768 and 1536 are usable truncations of its native 3072.
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))
EMBED_ID = f"{EMBED_MODEL}@{EMBED_DIM}"

# Gemini free tier, per Google project: roughly 100 requests/min, 30k tokens/min
# and 1,000 requests/day. Tokens per minute is the one ingest runs into — the
# corpus is ~1.4M tokens, so a full build takes ~45 minutes however it is
# batched. Raise EMBED_TPM on a paid tier.
EMBED_TPM = int(os.getenv("EMBED_TPM", "30000"))
# Texts per request: ~32 records is ~11k tokens, a third of a minute's budget.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", Path(__file__).parent / "chroma_db"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "compliance")

# Hugging Face dataset repo holding a built copy of CHROMA_DIR, e.g.
# "your-name/compliance-chroma". Set only where the index cannot be built in
# place — a Streamlit Cloud container, whose disk is empty on every cold start
# and where a 35-minute ingest is not an option. Unset locally, which is what
# keeps ingest.py from ever downloading over an index it is about to write.
INDEX_REPO = os.getenv("CHROMA_INDEX_REPO")


def hf_token():
    """The Hugging Face token, under either name this project uses.

    langchain's HuggingFaceEndpoint reads HUGGINGFACEHUB_API_TOKEN and
    huggingface_hub reads HF_TOKEN, so a working .env may define either one.
    Accepting both keeps the index download authenticated regardless of which
    name the deployment was configured with — an unauthenticated read of a
    private repo fails as a bare 404.
    """
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Records are capped at this many tokens. gemini-embedding-001 accepts 2,048, so this
# is no longer the model's limit but a chosen granularity: one record is one
# citable passage, short enough that the reranker scores the clause rather than
# the section around it, and that six of them fit the prompt with room for
# history. The splitter in ingest.py sizes itself against the same number.
EMBED_LIMIT = 512


@lru_cache(maxsize=1)
def get_tokenizer():
    """Token counter for the record cap and the free-tier pacing.

    Gemini's tokenizer is not available offline, so cl100k_base stands in. It
    tracks Gemini's counts closely on English prose, and nothing here needs an
    exact number: the cap is a chosen granularity well under Gemini's 2,048-token
    input limit, and pacing overshoot is caught by the 429 retry below. Still
    far better than the corpus's own token_estimate, which is char_count/4.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def token_length(text):
    return len(get_tokenizer().encode(text, disallowed_special=()))


def gemini_api_key():
    """Gemini key under either name the google-genai SDK reads."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


class GeminiEmbeddings(Embeddings):
    """gemini-embedding-001 as a LangChain embedder, paced for the free tier.

    Asymmetric, like BGE was: passages are embedded as RETRIEVAL_DOCUMENT and
    questions as RETRIEVAL_QUERY, which places a question near the passages
    that answer it rather than near other questions.

    Vectors are L2-normalised here. Only the native 3072 come back unit-length;
    a truncated 768 or 1536 does not, and cosine distance on unnormalised
    vectors is not cosine similarity.
    """

    def __init__(self, model=EMBED_MODEL, dim=EMBED_DIM, tokens_per_minute=EMBED_TPM):
        from google import genai

        key = gemini_api_key()
        if not key:
            raise RuntimeError(
                "No GOOGLE_API_KEY (or GEMINI_API_KEY) set. Create one at "
                "https://aistudio.google.com/apikey and add it to .env."
            )
        self.model, self.dim, self.tpm = model, dim, tokens_per_minute
        self._client = genai.Client(api_key=key)
        # (timestamp, tokens) of requests sent in the last minute. Locked
        # because Streamlit serves sessions from several threads.
        self._sent = deque()
        self._lock = threading.Lock()

    def _pace(self, tokens):
        """Block until `tokens` more fit within the last minute's budget."""
        with self._lock:
            while True:
                now = time.monotonic()
                while self._sent and now - self._sent[0][0] >= 60:
                    self._sent.popleft()
                used = sum(n for _, n in self._sent)
                if not self._sent or used + tokens <= self.tpm:
                    self._sent.append((now, tokens))
                    return
                time.sleep(60 - (now - self._sent[0][0]) + 0.1)

    def _embed(self, texts, task_type):
        import numpy as np
        from google.genai import errors, types

        self._pace(sum(token_length(text) for text in texts))
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        for attempt in range(6):
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                break
            except errors.APIError as exc:
                # 429 is the free tier saying "slow down"; 5xx is transient.
                # Anything else — a bad key, a bad model name — will not get
                # better by waiting.
                retryable = exc.code == 429 or (exc.code or 0) >= 500
                if not retryable or attempt == 5:
                    if exc.code == 429:
                        raise RuntimeError(
                            "Gemini rate limit still exceeded after retries. If this "
                            "is the daily quota, resume tomorrow with "
                            "`python ingest.py build --resume`."
                        ) from exc
                    raise
                wait = min(90, 10 * 2**attempt)
                logging.warning("gemini embed %s, retrying in %ss", exc.code, wait)
                time.sleep(wait)

        vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.tolist()

    def embed_documents(self, texts):
        out = []
        for start in range(0, len(texts), EMBED_BATCH):
            out.extend(self._embed(texts[start : start + EMBED_BATCH], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


@lru_cache(maxsize=1)
def get_embeddings():
    """The one embedder both ingest and retrieval use. Needs GOOGLE_API_KEY."""
    return GeminiEmbeddings()


def _index_present():
    """Whether CHROMA_DIR holds a real index rather than an empty directory.

    Chroma creates the directory and its sqlite file on first connection, so
    "the path exists" is not the question — an empty index answers every query
    with nothing, and the failure looks like a bad corpus rather than a missing
    download.
    """
    sqlite = CHROMA_DIR / "chroma.sqlite3"
    return sqlite.exists() and sqlite.stat().st_size > 0


def ensure_index():
    """Fetch the prebuilt index from INDEX_REPO when there isn't one on disk.

    A no-op unless CHROMA_INDEX_REPO is set and the local index is missing, so
    the local workflow (ingest.py writes CHROMA_DIR, everything else reads it)
    is untouched. Downloads roughly 100MB, once per container.
    """
    if not INDEX_REPO or _index_present():
        return

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    token = hf_token()
    logging.info("no local index; fetching %s into %s", INDEX_REPO, CHROMA_DIR)
    try:
        snapshot_download(
            repo_id=INDEX_REPO,
            repo_type="dataset",
            local_dir=str(CHROMA_DIR),
            token=token,
        )
    except RepositoryNotFoundError as exc:
        # The Hub returns 404 rather than 403 for a private repo the caller
        # cannot see, so "missing" and "no access" are the same response here and
        # the raw error sends you looking for a repo that may exist. Spell out
        # the three real causes instead.
        raise RuntimeError(
            f"Cannot read the index repo {INDEX_REPO!r} (CHROMA_INDEX_REPO).\n"
            f"HF_TOKEN / HUGGINGFACEHUB_API_TOKEN is {'set' if token else 'NOT set'}. "
            "A private repo the token "
            "cannot see returns the same 404 as one that does not exist, so this is "
            "one of:\n"
            "  1. The index was never published. Run, from a local checkout with a "
            "built chroma_db/:\n"
            f"       HF_TOKEN=<write-token> python ingest.py push-index {INDEX_REPO}\n"
            "  2. CHROMA_INDEX_REPO still holds the placeholder from "
            "secrets.toml.example. Set it to a real <user>/<repo>.\n"
            "  3. HF_TOKEN belongs to an account without read access to it.\n"
            "Unset CHROMA_INDEX_REPO to use a local chroma_db/ instead."
        ) from exc


class EmbeddingModelMismatch(RuntimeError):
    """The collection on disk was built with a different embedding model."""


@lru_cache(maxsize=None)
def get_store(check_model=True):
    """Handle on the persistent collection.

    Cached because Streamlit re-runs the whole script on every interaction, and a
    fresh Chroma client per rerun means a new sqlite connection and a reloaded
    HNSW index each time. Anything that deletes the collection must call
    get_store.cache_clear(), or it keeps talking to a handle that no longer exists.

    hnsw:space is frozen when the collection is first created and cannot be
    changed afterwards — Chroma's default is l2, which would be the wrong metric
    for the normalized vectors above. Changing it later means a full rebuild.

    The embedding model is frozen the same way and recorded beside it, so an
    index built with another model fails here, by name, instead of answering
    every question with unrelated passages. check_model=False is for the one
    caller that must open a stale collection: ingest.py, to delete it.
    """
    ensure_index()
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "embed_model": EMBED_ID},
        # Off because chromadb 0.5.23 calls the installed posthog with a
        # signature it no longer accepts, printing a telemetry error per batch.
        client_settings=Settings(anonymized_telemetry=False),
    )
    # Chroma returns an existing collection as-is, ignoring the metadata above,
    # so this reads what the collection was actually built with.
    built_with = (store._collection.metadata or {}).get("embed_model")
    if check_model and built_with != EMBED_ID:
        raise EmbeddingModelMismatch(
            f"Collection {COLLECTION!r} in {CHROMA_DIR} was embedded with "
            f"{built_with or 'an unrecorded model (BAAI/bge-base-en-v1.5)'}, "
            f"but EMBED_MODEL/EMBED_DIM say {EMBED_ID!r}. Rebuild it:\n"
            "    python ingest.py build --reset"
        )
    return store


def build_filter(framework=None, sector=None):
    """Chroma `where` clause over the corpus's two filter keys.

    Chroma rejects a multi-key dict, so anything past one term needs $and.
    Returns None when unfiltered, which is what as_retriever expects.
    """
    terms = []
    if framework:
        terms.append({"framework_id": framework})
    if sector:
        terms.append({"sector": sector})
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else {"$and": terms}


def get_retriever(k=5, framework=None, sector=None):
    """A standard LangChain retriever, ready to drop into a RAG chain."""
    kwargs = {"k": k}
    where = build_filter(framework, sector)
    if where:
        kwargs["filter"] = where
    return get_store().as_retriever(search_kwargs=kwargs)


@lru_cache(maxsize=1)
def indexed_frameworks():
    """Framework ids actually present in the index, for scoping a search.

    Read from the collection rather than from the corpus on disk so the UI can
    only offer filters that will actually match something.
    """
    metas = get_store().get(include=["metadatas"])["metadatas"]
    return sorted({m["framework_id"] for m in metas})


def display_text(document):
    """The clean clause text, without the embedding header.

    page_content is embed_text — framework/regulator/date and breadcrumb prefixed
    onto the clause — because that is what gets embedded. Showing that header to
    a reader is noise, so it is sliced back off using the length recorded at
    ingest time. The corpus README's rule: embed embed_text, show text.
    """
    return document.page_content[document.metadata.get("header_chars", 0) :]
