# Compliance Chat

A retrieval-augmented assistant over Indian and international regulatory
frameworks. It answers compliance questions **only** from an indexed corpus of
primary regulation, cites the section behind every claim, and refuses when the
corpus does not cover the question.

**Live app:** https://rag-based-compliance-chatbot.streamlit.app

The design goal is narrow and worth stating plainly: an assistant that answers
compliance questions from a language model's memory is worse than no assistant,
because a confident wrong answer about a regulatory obligation is acted upon. So
the model here never answers from memory. Every claim traces to a retrieved
passage, and when retrieval comes back empty the system says so instead of
improvising.

## What is indexed

**27 documents · 25 frameworks · 2,602 source chunks → 4,060 indexed records.**

| Sector | Frameworks |
| --- | --- |
| Financial | RBI (KYC, Cybersecurity, Digital Lending, Account Aggregator, IT Outsourcing), SEBI (Cybersecurity, AML/CFT, Investment Adviser, Algo Trading), IRDAI (Cyber Security, AML/CFT, Web Aggregator), PCI-DSS v4, PMLA |
| Universal | DPDP Act 2023, DPDP Rules 2025, IT Act 2000, IT Rules 2011, CERT-In Directions 2022 |
| Corporate | Companies Act 2013, Labour Codes 2020, Consumer Protection (E-Commerce) Rules 2020, Telemedicine Guidelines 2020 |
| Global | GDPR, ISO 27001:2022 |

Chunks carry framework, regulator, breadcrumb and page number, so a citation
points at something you can check against the source PDF.

## How it works

```
question
   │
   ├─ condense ............ rewrites a follow-up into a standalone question,
   │                        so "what about GDPR?" retrieves on more than four words
   ├─ input guard ......... length limits, prompt-injection framing
   ├─ retrieve (k=24) ..... BGE embeddings, cosine distance over Chroma
   ├─ distance floor ...... anything past 0.45 is dropped before the model sees it
   ├─ diversify ........... max 2 chunks per section, so one long section cannot
   │                        consume the whole context budget
   └─ grounded prompt ..... numbered sources, inline [n] citations, or a refusal
```

Three guardrails do the real work:

**A distance floor.** Vector search always returns *k* results, however bad the
match — an off-topic question still gets its nearest neighbours back at distance
0.59, while a genuine hit lands around 0.23. Without a floor the model receives
irrelevant regulation and writes a confident answer around it. Nothing past 0.45
reaches the model.

**Citation validation.** The model is told to cite inline, and citations are
checked against the sources actually supplied — a reference to a source number
outside the provided range is caught rather than rendered.

**Refusal that means refusal.** "This is out of my knowledge" is reserved for the
case where the sources genuinely do not answer, and is stripped when the model
tacks it onto an answer it *did* support. The UI counts only the passages an
answer actually cited, because calling a merely-retrieved passage a "source"
claims it backed the answer when it did not.

## Retrieval quality

`python eval_retrieval.py` measures retrieval alone — no model calls, no cost,
runs in under a minute:

```
30 cases  (k=6, floor=0.45, fetch=24)

answerable   24/26  (92% of questions retrieved the passage that answers them)
refused      4/4    (100% of out-of-corpus questions correctly returned nothing)
mean rank    1.4    (where the answering passage landed, 1 is best)
```

Retrieval is the ceiling on the whole system: a passage that never surfaces
cannot be cited however good the prompt is. Each case names a phrase the corpus
actually contains, verified against the corpus rather than inferred from what
retrieval happened to return.

## Running it locally

Requires Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env        # add HF_TOKEN
python ingest.py build      # ~35 minutes; builds chroma_db/
streamlit run app.py
```

The index is not in the repo — it is ~99MB of generated artefact. Build it
locally, or point `CHROMA_INDEX_REPO` at a published copy and it will be fetched
on first use.

### CLI

```bash
python ingest.py stats                    # what is indexed, and whether it is complete
python ingest.py query "KYC periodic update"   # search the index directly
python eval_retrieval.py --verbose        # retrieval quality
python store.py list                      # saved conversations
pytest                                    # 39 tests
```

## Layout

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Streamlit UI — chat, history sidebar, framework scoping, citations |
| [`rag.py`](rag.py) | Retrieval policy, guardrails, prompt construction |
| [`vectorstore.py`](vectorstore.py) | Chroma + BGE embeddings; the single source of truth for how documents are embedded |
| [`chat.py`](chat.py) | Model construction and LangSmith run configuration |
| [`store.py`](store.py) | SQLite chat history |
| [`ingest.py`](ingest.py) | Corpus → index, plus `stats` / `query` / `push-index` |
| [`corpus_repair.py`](corpus_repair.py) | Fixes applied to source chunks at ingest time |
| [`eval_retrieval.py`](eval_retrieval.py) | Retrieval evaluation harness |

`vectorstore.py`, `store.py` and `rag.py` deliberately import no Streamlit, so
the retrieval policy stays testable from the CLI and reusable behind an API.

## Deployment

See [DEPLOY.md](DEPLOY.md). The short version: the index is built locally,
published to a Hugging Face dataset repo, and downloaded once per container —
a Streamlit Cloud container starts with an empty disk, and a 35-minute ingest is
not something you can do at startup.

## Limitations

- **Not legal advice.** It reports what the indexed text says. Regulation changes,
  and the corpus is a snapshot.
- **The corpus is the boundary.** Anything outside those 25 frameworks is refused
  by design, including questions a general-purpose model could answer.
- **Chat history is not durable on Streamlit Cloud.** The container is ephemeral,
  so conversations are lost on restart. Local runs persist normally.
