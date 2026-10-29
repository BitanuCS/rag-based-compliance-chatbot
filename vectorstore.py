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

# Cohere embedding model and output size. Both are stamped onto the collection
# when it is created and checked on every open: vectors from two models (or two
# sizes of one) live in different spaces, and querying one with the other
# returns noise rather than an error. embed-v4.0 offers 256/512/1024/1536.
EMBED_MODEL = os.getenv("EMBED_MODEL", "embed-v4.0")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
EMBED_ID = f"{EMBED_MODEL}@{EMBED_DIM}"

# Texts per embed call: Cohere's maximum. The trial key's budget is 1,000 calls
# a month across every endpoint, counted per call rather than per text — so a
# full ~4,300-chunk build is ~45 calls, and batching as wide as allowed is what
# makes the free tier enough. (Gemini's free tier, tried first, counts every
# text as a request and caps at 1,000 a day: five days for one build.)
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "96"))

# Tokens per minute sent to the embedder. A trial key is limited to 100,000
# (the 429 body says so), and a 96-chunk batch is ~32k, so a build runs ~3
# batches a minute — ~15 minutes for the corpus. Paced 20% under the limit
# because the count here is cl100k, not Cohere's own tokenizer, and a rejected
# call may still count against the monthly 1,000. Raise on a production key.
EMBED_TPM = int(os.getenv("EMBED_TPM", "80000"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", Path(__file__).parent / "chroma_db"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "compliance")

# Hugging Face dataset repo holding a built copy of CHROMA_DIR, e.g.
# "your-name/compliance-chroma". Set only where the index cannot be built in
# place — a Streamlit Cloud container, whose disk is empty on every cold start
# and where a 35-minute ingest is not an option. Unset locally, which is what
# keeps pdf_to_chromadb.py from ever downloading over an index it is about to write.
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

# Records are capped at this many tokens. embed-v4.0 accepts far more, so this
# is no longer the model's limit but a chosen granularity: one record is one
# citable passage, short enough that the reranker scores the clause rather than
# the section around it, and that six of them fit the prompt with room for
# history. The chunker in pdf_to_chromadb.py sizes itself against the same number.
EMBED_LIMIT = 512


@lru_cache(maxsize=1)
def get_tokenizer():
    """Token counter for the 512-token record cap.

    cl100k_base stands in for Cohere's tokenizer, which is not available
    offline. Nothing here needs Cohere's exact count: embed-v4.0 accepts far
    longer inputs than 512 tokens, so the cap is a chosen granularity, not a
    truncation cliff. Still far better than the corpus's own token_estimate,
    which was char_count/4.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def token_length(text):
    return len(get_tokenizer().encode(text, disallowed_special=()))


def cohere_api_key():
    """Cohere key under either name: COHERE_API_KEY, or the SDK's own CO_API_KEY."""
    return os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")


# Rate-limit and transient-failure retries for one embed call.
EMBED_RETRIES = 6


class CohereEmbeddings(Embeddings):
    """Cohere embed-v4.0 as a LangChain embedder.

    Asymmetric, as BGE was: passages are embedded as search_document and
    questions as search_query, which places a question near the passages that
    answer it rather than near other questions. Vectors are L2-normalised here
    so cosine distance in Chroma is cosine similarity, whatever the model sends.
    """

    def __init__(self, model=EMBED_MODEL, dim=EMBED_DIM, tokens_per_minute=EMBED_TPM):
        import cohere

        key = cohere_api_key()
        if not key:
            raise RuntimeError(
                "No COHERE_API_KEY (or CO_API_KEY) set. Create one at "
                "https://dashboard.cohere.com/api-keys and add it to .env."
            )
        self.model, self.dim, self.tpm = model, dim, tokens_per_minute
        self._client = cohere.ClientV2(api_key=key, timeout=60, max_retries=0)
        # (timestamp, tokens) of calls in the last minute. Locked because
        # Streamlit serves sessions from several threads.
        self._sent = deque()
        self._lock = threading.Lock()

    def _pace(self, tokens):
        """Block until `tokens` more fit within the last minute's budget."""
        with self._lock:
            while True:
                now = time.monotonic()
                while self._sent and now - self._sent[0][0] >= 60:
                    self._sent.popleft()
                if not self._sent or sum(n for _, n in self._sent) + tokens <= self.tpm:
                    self._sent.append((now, tokens))
                    return
                time.sleep(60 - (now - self._sent[0][0]) + 0.1)

    def _embed(self, texts, input_type):
        import cohere
        import numpy as np

        self._pace(sum(token_length(text) for text in texts))
        for attempt in range(EMBED_RETRIES):
            try:
                response = self._client.embed(
                    model=self.model,
                    input_type=input_type,
                    texts=texts,
                    embedding_types=["float"],
                    output_dimension=self.dim,
                )
                break
            except (
                cohere.TooManyRequestsError,
                cohere.ServiceUnavailableError,
                cohere.InternalServerError,
                cohere.GatewayTimeoutError,
            ) as exc:
                # A trial key's per-minute limit, or a transient outage: wait
                # and retry. Anything else — a bad key, a bad model name — will
                # not get better by waiting, and raises at once.
                if attempt == EMBED_RETRIES - 1:
                    raise
                wait = min(90, 10 * 2**attempt)
                logging.warning("cohere embed %s, retrying in %ss", type(exc).__name__, wait)
                time.sleep(wait)

        vectors = np.array(response.embeddings.float_, dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.tolist()

    def embed_documents(self, texts):
        out = []
        for start in range(0, len(texts), EMBED_BATCH):
            out.extend(self._embed(texts[start : start + EMBED_BATCH], "search_document"))
        return out

    def embed_query(self, text):
        return self._embed([text], "search_query")[0]


@lru_cache(maxsize=1)
def get_embeddings():
    """The one embedder both the pipeline and retrieval use. Needs COHERE_API_KEY."""
    return CohereEmbeddings()


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
    the local workflow (pdf_to_chromadb.py writes CHROMA_DIR, everything else reads it)
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
            f"       HF_TOKEN=<write-token> python pdf_to_chromadb.py push-index {INDEX_REPO}\n"
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
    caller that must open a stale collection: pdf_to_chromadb.py, to delete it.
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
            "    python pdf_to_chromadb.py build --reset"
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
