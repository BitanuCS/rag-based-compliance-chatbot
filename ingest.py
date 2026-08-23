"""Load the compliance corpus, split what is too long, and store it in Chroma.

The corpus arrives pre-chunked: 2,602 chunks, each carrying an `embed_text`
field that prefixes a `[framework | regulator | date]` line and a breadcrumb
onto the clause text. That header is the point — it encodes *where* a clause
sits, not only what it says — so this pipeline preserves it rather than
re-cutting the corpus from scratch.

What it does not preserve is oversized chunks: measured with the embedder's own
tokenizer, 1,271 of the 2,602 (48.8%) run past bge-base's 512-token input limit,
the largest at 2,724 tokens. Embedded as-is, everything past token 512 is
silently dropped and becomes unretrievable. So the split step is a guarded
repair, not a re-chunk: chunks that fit pass through untouched, and only the
oversized ones are split — with the header re-applied to every part so each part
still says which framework and section it came from.

    python ingest.py build [--reset] [--frameworks ID,ID] [--limit N]
    python ingest.py stats
    python ingest.py query "..." [--framework ID] [--sector S] [--k 5]
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import corpus_repair

from vectorstore import (
    CHROMA_DIR,
    _index_present,
    COLLECTION,
    EMBED_LIMIT,
    EMBED_MODEL,
    display_text,
    get_embeddings,
    get_store,
    get_tokenizer,
    indexed_frameworks,
    token_length,
)

CORPUS = Path(__file__).parent / "data" / "all_chunks.json"
VALIDATION = Path(__file__).parent / "data" / "validation.json"

# Body budget per part. The largest header in the corpus is 88 tokens; 2 more go
# to the [CLS]/[SEP] pair. A single fixed budget keeps one splitter instance for
# the whole run, and costs at most ~30 tokens against a per-chunk budget.
MAX_HEADER_TOKENS = 88
BODY_BUDGET = EMBED_LIMIT - MAX_HEADER_TOKENS - 2
OVERLAP = 64

# Progress granularity and MPS memory headroom, not a Chroma limit.
BATCH_SIZE = 256

# Copied onto every record. framework_id and sector are the filter keys; the
# rest are for citation and provenance.
CARRY = (
    "chunk_id",
    "framework_id",
    "framework_name",
    "regulator",
    "sector",
    "effective_date",
    "document_status",
    "section_id",
    "section_number",
    "section_title",
    "section_path",
    "chapter",
    "breadcrumb",
    "page_start",
    "page_end",
    "source_url",
    "source_pdf",
)


def load_chunks(frameworks=None, limit=None):
    """Read all_chunks.json, optionally narrowed to some frameworks.

    all_chunks.json is the union of the 27 per-framework chunk files, so it is
    the whole corpus in one read.
    """
    chunks = json.loads(CORPUS.read_text())
    if frameworks:
        wanted = set(frameworks)
        unknown = wanted - {c["framework_id"] for c in chunks}
        if unknown:
            sys.exit(f"unknown framework ids: {', '.join(sorted(unknown))}")
        chunks = [c for c in chunks if c["framework_id"] in wanted]
    return chunks[:limit] if limit else chunks


def header_of(chunk):
    """The embedding header that embed_text prefixes onto text.

    Slicing is exact and holds for all 2,602 chunks. The assertion is what makes
    a future corpus revision fail loudly here instead of quietly indexing text
    with a mangled header.
    """
    embed_text, text = chunk["embed_text"], chunk["text"]
    if not embed_text.endswith(text):
        raise ValueError(
            f"{chunk['chunk_id']}: embed_text does not end with text; "
            "the header convention this splitter relies on has changed"
        )
    return embed_text[: len(embed_text) - len(text)]


def clean_metadata(chunk, header, index, total, overrides=None):
    """Corpus fields reduced to what Chroma accepts.

    Chroma stores str/int/float/bool only. Two fields in this corpus break that:
    `chapter` is None on most chunks, and `section_path` is a list.

    `overrides` carries corrections from corpus_repair — a lifted schedule keeps
    its parent's provenance (framework, pages, source URL) but gets its own
    section identity, which is the whole point of lifting it out.
    """
    meta = {}
    for key in CARRY:
        value = chunk.get(key)
        if value is None:  # `chapter`, mostly — Chroma rejects None outright
            continue
        meta[key] = "/".join(map(str, value)) if isinstance(value, list) else value

    # Length of the header inside page_content, so readers can slice it back off
    # instead of us storing all 13MB of clause text a second time.
    meta["header_chars"] = len(header)
    # Split parts trace back to one citable section.
    meta["parent_chunk_id"] = chunk["chunk_id"]
    meta["split_index"] = index
    meta["split_total"] = total
    if overrides:
        # A schedule sits beside the chapter it was printed after, not inside it.
        meta.pop("chapter", None)
        meta.update(overrides)
    return meta


def build_documents(chunks, splitter):
    """Chunks in, (documents, ids) out — splitting only what overruns the model.

    Ids are deterministic (`chunk_id`, `chunk_id#schedule`, or `…#pN` for parts)
    so a re-run upserts in place rather than duplicating the corpus.
    """
    documents, ids = [], []
    passed = split = repaired = 0

    for chunk in chunks:
        header = header_of(chunk)

        # A chunk normally yields itself. One carrying a schedule yields the body
        # and the schedule separately, so the schedule stops answering to the
        # breadcrumb of whichever section it was printed after.
        for suffix, text, overrides in corpus_repair.repair(chunk):
            part_header = header
            if "breadcrumb" in overrides:
                part_header = header.replace(chunk["breadcrumb"], overrides["breadcrumb"], 1)
                repaired += 1
            embed_text = part_header + text
            record_id = chunk["chunk_id"] + suffix

            if token_length(embed_text) <= EMBED_LIMIT:
                documents.append(
                    Document(
                        page_content=embed_text,
                        metadata=clean_metadata(chunk, part_header, 1, 1, overrides),
                    )
                )
                ids.append(record_id)
                passed += 1
                continue

            # Split the body alone, then re-apply the header to each part so every
            # part remains self-describing.
            parts = splitter.split_text(text)
            for i, part in enumerate(parts, start=1):
                documents.append(
                    Document(
                        page_content=part_header + part,
                        metadata=clean_metadata(chunk, part_header, i, len(parts), overrides),
                    )
                )
                ids.append(f"{record_id}#p{i}")
            split += 1

    return documents, ids, passed, split, repaired


def build(frameworks=None, limit=None, reset=False, assume_yes=False, replace=False):
    chunks = load_chunks(frameworks, limit)
    print(f"loaded {len(chunks)} chunks from {CORPUS.name}")

    if replace:
        if not frameworks:
            sys.exit("--replace needs --frameworks; use --reset to drop everything")
        # Ids are deterministic, so re-running upserts — but only over the ids the
        # new run happens to produce. A chunk that used to be split into #p1/#p2
        # and now fits in one record would leave those parts behind as orphans
        # answering to text that no longer exists. Clearing the framework first is
        # what makes a targeted re-ingest equivalent to a rebuild of that slice.
        store = get_store()
        before = store._collection.count()
        store._collection.delete(where={"framework_id": {"$in": list(frameworks)}})
        removed = before - store._collection.count()
        print(f"replace: removed {removed} existing records for {', '.join(frameworks)}")

    if reset:
        store = get_store()
        existing = store._collection.count()
        if existing and not assume_yes:
            answer = input(f"delete collection {COLLECTION!r} ({existing} records)? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                sys.exit("aborted")
        store.delete_collection()
        # The cached handle now points at a collection that no longer exists.
        get_store.cache_clear()
        indexed_frameworks.cache_clear()
        print(f"reset collection {COLLECTION!r}")

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        get_tokenizer(),
        chunk_size=BODY_BUDGET,
        chunk_overlap=OVERLAP,
        # Legal prose degrades gracefully: paragraph, line, sentence, clause.
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    documents, ids, passed, split, repaired = build_documents(chunks, splitter)
    print(
        f"split: {passed} chunks fit as-is, {split} were over {EMBED_LIMIT} tokens "
        f"→ {len(documents)} records"
    )
    if repaired:
        print(f"repair: {repaired} schedules lifted out of the section they were printed under")

    oversized = [d for d in documents if token_length(d.page_content) > EMBED_LIMIT]
    if oversized:
        sys.exit(f"{len(oversized)} records still exceed {EMBED_LIMIT} tokens; not indexing")

    print(f"embedding with {EMBED_MODEL} on {get_embeddings().model_kwargs['device']}...")
    store = get_store()
    for start in range(0, len(documents), BATCH_SIZE):
        window = slice(start, start + BATCH_SIZE)
        store.add_documents(documents[window], ids=ids[window])
        done = min(start + BATCH_SIZE, len(documents))
        print(f"  {done}/{len(documents)}", flush=True)

    print(f"\nstored {len(documents)} records in {CHROMA_DIR}")


def stats():
    """Record counts, per-framework parity against the corpus, truncation check."""
    store = get_store()
    data = store.get(include=["metadatas", "documents"])
    metas, docs = data["metadatas"], data["documents"]

    print(f"{CHROMA_DIR}  collection={COLLECTION!r}  model={EMBED_MODEL}")
    print(f"\n{len(metas)} records from {len({m['parent_chunk_id'] for m in metas})} source chunks")
    if not metas:
        return

    parts = Counter(m["framework_id"] for m in metas)
    parents = Counter()
    for meta in metas:
        # Repaired records are lifted out of a parent, not parents themselves, so
        # counting them would show more chunks than the corpus contains.
        if meta.get("repair"):
            continue
        parents[meta["framework_id"]] += meta["split_index"] == 1

    # The corpus records its own per-framework chunk counts; anything short of
    # them means chunks were dropped somewhere in the pipeline.
    expected = json.loads(VALIDATION.read_text())["chunks_per_framework"]
    print(f"\n{'framework':<42} {'chunks':>7} {'expected':>9} {'records':>8}")
    missing = []
    for framework, want in sorted(expected.items(), key=lambda kv: -kv[1]):
        got = parents.get(framework, 0)
        flag = "" if got == want else "  ✗"
        if got != want:
            missing.append(framework)
        print(f"{framework:<42} {got:>7} {want:>9} {parts.get(framework, 0):>8}{flag}")

    lifted = sum(1 for m in metas if m.get("repair") == "schedule")
    print(f"\nschedules lifted into their own records: {lifted}")

    over = sum(1 for d in docs if token_length(d) > EMBED_LIMIT)
    print(f"records over {EMBED_LIMIT} tokens: {over}" + ("  ✗" if over else "  ✓"))
    if missing:
        print(f"frameworks not fully indexed: {', '.join(missing)}  ✗")
    elif not over:
        print("all frameworks fully indexed  ✓")


def query(text, k=5, framework=None, sector=None):
    from vectorstore import build_filter

    store = get_store()
    where = build_filter(framework, sector)
    hits = store.similarity_search_with_score(text, k=k, filter=where)

    scope = framework or sector or "all frameworks"
    print(f"query: {text!r}  ({scope}, k={k})\n")
    for rank, (doc, distance) in enumerate(hits, start=1):
        meta = doc.metadata
        pages = f"p{meta['page_start']}"
        if meta["page_end"] != meta["page_start"]:
            pages += f"-{meta['page_end']}"
        part = ""
        if meta["split_total"] > 1:
            part = f" [part {meta['split_index']}/{meta['split_total']}]"
        body = " ".join(display_text(doc).split())
        print(f"{rank}. {meta['framework_id']}  {pages}  cosine={distance:.4f}{part}")
        print(f"   {meta['breadcrumb']}")
        print(f"   {body[:220]}...\n")


def push_index(repo_id, private=True):
    """Upload the built index to a Hugging Face dataset repo.

    The deployed app has no way to build an index — a container starts with an
    empty disk and a full ingest is ~35 minutes — so the built artefact is
    published here and fetched by vectorstore.ensure_index() at startup. Run
    this after any rebuild, or the deployment keeps serving the old index.

    A dataset repo rather than a model repo: this is data, and dataset repos are
    what the Hub's own tooling expects for a blob of files like this.
    """
    from huggingface_hub import HfApi

    if not _index_present():
        sys.exit(f"no index at {CHROMA_DIR} — run `ingest.py build` first")

    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set; needs a token with write permission")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"uploading {CHROMA_DIR} to {repo_id} (dataset)…")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(CHROMA_DIR),
        commit_message="rebuild compliance index",
    )
    print(f"done. Set CHROMA_INDEX_REPO={repo_id} in the deployment's secrets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="load, split and index the corpus")
    build_cmd.add_argument("--reset", action="store_true", help="drop the collection first")
    build_cmd.add_argument("--yes", action="store_true", help="skip the --reset confirmation")
    build_cmd.add_argument("--frameworks", help="comma-separated framework ids to index")
    build_cmd.add_argument(
        "--replace",
        action="store_true",
        help="delete the selected frameworks' records first (targeted re-ingest)",
    )
    build_cmd.add_argument("--limit", type=int, help="index only the first N chunks (smoke test)")

    sub.add_parser("stats", help="what is indexed, and whether it is complete")

    push_cmd = sub.add_parser("push-index", help="upload the built index to a HF dataset repo")
    push_cmd.add_argument("repo_id", help='e.g. "your-name/compliance-chroma"')
    push_cmd.add_argument(
        "--public", action="store_true", help="create the repo public (default: private)"
    )

    query_cmd = sub.add_parser("query", help="search the index")
    query_cmd.add_argument("text")
    query_cmd.add_argument("--k", type=int, default=5)
    query_cmd.add_argument("--framework")
    query_cmd.add_argument("--sector")

    args = parser.parse_args()

    if args.command == "build":
        frameworks = args.frameworks.split(",") if args.frameworks else None
        build(frameworks, args.limit, args.reset, args.yes, args.replace)
    elif args.command == "stats":
        stats()
    elif args.command == "push-index":
        push_index(args.repo_id, private=not args.public)
    elif args.command == "query":
        query(args.text, args.k, args.framework, args.sector)


if __name__ == "__main__":
    main()
