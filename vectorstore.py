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
from functools import lru_cache
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Silences a "fork after parallelism" warning on every batch: sentence-transformers
# tokenizes in the parent, then Chroma's client forks underneath it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# chromadb 0.5.23 calls posthog.capture() with an argument list posthog 7.x no
# longer accepts, so it logs a failure per client even with telemetry off below.
# Nothing is sent either way; this just keeps the noise out of the CLI output.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", Path(__file__).parent / "chroma_db"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "compliance")

# bge-base truncates silently past this. The splitter in ingest.py sizes itself
# against the same number.
EMBED_LIMIT = 512

# BGE is trained for asymmetric retrieval: the instruction goes on the query so
# it lands near passages that answer it, and never on the passages themselves.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _device():
    """Apple Silicon GPU when torch can see it, else CPU."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class BgeEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings with BGE's query instruction.

    The modern langchain_huggingface class has no query_instruction field — only
    the deprecated community HuggingFaceBgeEmbeddings does — so the prefix is
    applied here. embed_documents is deliberately left alone.
    """

    def embed_query(self, text):
        return super().embed_query(QUERY_INSTRUCTION + text)


@lru_cache(maxsize=1)
def get_tokenizer():
    """The embedder's own tokenizer, so lengths are measured in real tokens.

    The corpus ships a token_estimate field, but it is char_count/4 and drifts
    far enough from the truth to matter at a hard 512-token cliff.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def token_length(text):
    return len(get_tokenizer()(text)["input_ids"])


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the sentence-transformer once per process (it is slow to build)."""
    return BgeEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": _device()},
        # Required, not cosmetic: cosine distance on unnormalized vectors is not
        # cosine similarity.
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_store():
    """Handle on the persistent collection.

    Cached because Streamlit re-runs the whole script on every interaction, and a
    fresh Chroma client per rerun means a new sqlite connection and a reloaded
    HNSW index each time. Anything that deletes the collection must call
    get_store.cache_clear(), or it keeps talking to a handle that no longer exists.

    hnsw:space is frozen when the collection is first created and cannot be
    changed afterwards — Chroma's default is l2, which would be the wrong metric
    for the normalized BGE vectors above. Changing it later means a full rebuild.
    """
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
        # Off because chromadb 0.5.23 calls the installed posthog with a
        # signature it no longer accepts, printing a telemetry error per batch.
        client_settings=Settings(anonymized_telemetry=False),
    )


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
