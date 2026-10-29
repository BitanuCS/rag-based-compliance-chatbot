<div align="center">

# ⚖️ Compliance Chat

**A retrieval-augmented assistant over Indian and international regulation that
cites every claim — or refuses.**

[![Live app](https://img.shields.io/badge/demo-live-success?logo=streamlit&logoColor=white)](https://rag-based-compliance-chatbot.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](.python-version)
[![Frameworks](https://img.shields.io/badge/frameworks-25-informational)](#whats-indexed)
[![Retrieval](https://img.shields.io/badge/hit%401-88%25-brightgreen)](#retrieval-quality)
[![Tests](https://img.shields.io/badge/tests-68-brightgreen)](tests/)

[Live app](https://rag-based-compliance-chatbot.streamlit.app) ·
[Deployment](DEPLOY.md) ·
[Engineering notes](PHASES.md)

</div>

---

An assistant that answers compliance questions from a language model's memory is
worse than no assistant, because a confident wrong answer about a regulatory
obligation gets acted on. So this one never answers from memory. Every claim
traces to a retrieved passage with a framework, section and page you can check
against the source PDF — and when the corpus does not cover the question, it says
so instead of improvising.

## The bug that shaped the design

Asked whether a Data Principal can be penalised under the DPDP Act, an early
version answered that the Act *"does not specify a maximum penalty."*

It does — **up to ₹250 crore**, in the Schedule. The text had been indexed the
whole time.

The cause sat three layers below the model. The extraction pipeline never gave
the Schedule its own section, so the penalty table lived under a heading about
amendments to other Acts — and the heading is part of what gets embedded, so a
search for "maximum penalty" matched nothing. Worse, the same pipeline flattened
the table column by column: every breach description, then every serial number,
then every rupee amount. The breach → penalty mapping, which *was* the content,
was gone before it reached the vector store.

The answer looked fine. It cited its sources. It was wrong.

That is the failure mode this repo is built around: **retrieval quality is the
ceiling on the whole system**, and the interesting bugs are upstream of the
prompt. So the project now owns the whole path from the PDFs:
[`pdf_to_chromadb.py`](pdf_to_chromadb.py) gives the Schedule its own section, and rebuilds
gazette tables that have no ruling lines from the `(1) (2) (3)` column-number
row they print — so each breach is read next to its penalty, one row at a time.

## Features

- **Cited answers.** Every claim carries an inline `[n]`; a **References**
  list under the answer gives framework, section, pages and a link to the
  source document, one line per location.
- **Checkable evidence.** *Show cited passages* opens the exact text the model
  read, with its cosine distance and rerank relevance score. Only passages the
  answer actually cites are listed.
- **Refuses instead of guessing.** Out-of-corpus questions get a fixed refusal,
  usually with no model call at all.
- **Invalid-citation warning.** A `[n]` pointing at a source that was never
  supplied is flagged under the answer as unverified.
- **Search scope.** Restrict retrieval to one framework from the sidebar, or let
  the app detect the framework a question names.
- **Follow-up questions.** "What about GDPR?" is rewritten into a standalone
  question when it finds nothing on its own.
- **Saved chat history.** Conversations persist in SQLite; reopen or delete
  them from the sidebar.
- **Feedback.** 👍/👎 under each answer when tracing is on, saved with the chat
  and scored onto the turn's Langfuse trace.
- **Tracing.** Every turn traced in Langfuse and LangSmith (see
  [Observability](#observability)).

## Tech stack

| Layer | Choice |
| --- | --- |
| UI | Streamlit 1.48 |
| LLM | Hugging Face Inference API via `langchain-huggingface` — `HF_MODEL`, default `Qwen/Qwen2.5-7B-Instruct` (`.env.example` sets `meta-llama/Llama-3.3-70B-Instruct`) |
| Embeddings | Cohere `embed-v4.0`, 1536 dimensions |
| Reranker | Cohere `rerank-v3.5` |
| Vector store | Chroma 0.5 (cosine), published to a Hugging Face dataset for deployment |
| PDF extraction | PyMuPDF; Tesseract OCR for scanned pages |
| Token counting | `tiktoken` (`cl100k_base`) |
| Chat history | SQLite |
| Observability | Langfuse 5 and LangSmith |
| Tests | pytest |

## What's indexed

**27 documents · 25 frameworks · 2,816 sections → 4,299 chunks**

| Sector | Frameworks |
| --- | --- |
| **Financial** | RBI (KYC, Cybersecurity, Digital Lending, Account Aggregator, IT Outsourcing) · SEBI (Cybersecurity, AML/CFT, Investment Adviser, Algo Trading) · IRDAI (Cyber Security, AML/CFT, Web Aggregator) · PCI-DSS v4 · PMLA |
| **Universal** | DPDP Act 2023 · DPDP Rules 2025 · IT Act 2000 · IT Rules 2011 · CERT-In Directions 2022 |
| **Corporate** | Companies Act 2013 · Labour Codes 2020 · Consumer Protection (E-Commerce) Rules 2020 · Telemedicine Guidelines 2020 |
| **Global** | GDPR · ISO 27001:2022 |

## From PDF to index

```
source_pdfs/*.pdf
   │
   ├─ extract ............ PyMuPDF text; Tesseract OCR for the 6 scanned PDFs
   │                       drops running headers, footnotes, contents pages;
   │                       tables read row by row; margin notes → section titles
   ├─ structure .......... detects the document's heading style — gazette
   │                       "12. Title.—", "Article 12" or decimal "7.5.2 Title"
   ├─ chunk .............. one chunk per section, split past 512 tokens on
   │                       paragraph → sentence → clause, 64-token overlap
   ├─ data/*.json ........ the corpus, regenerated — never edited by hand
   └─ embed + store ...... Cohere embed-v4.0 → Chroma (cosine)
```

**Why structure-aware chunking.** A regulation is already chunked by its author:
one section is one obligation, and it is cited by its number. Cutting anywhere
else either splits a rule from its proviso or glues two unrelated rules
together, and the reader loses the "Section 8(5)" they need to check the
answer. Every chunk also carries a `[framework | regulator | date]` header and
its `Framework > Chapter > Section` breadcrumb into the embedding, because a
clause is only findable by what it is about if the embedding knows where it
sits.

## How it works

```
question
   │
   ├─ input guard ........ length cap; the question is data, never instructions
   ├─ framework detect ... "the DPDP Act" scopes to the Act, bare "DPDP" scopes
   │                       to neither, two frameworks named means a comparison
   ├─ condense ........... only if the raw question finds nothing and there is
   │                       history: rewrites "what about GDPR?" into a
   │                       standalone question and retrieves again
   ├─ retrieve ........... fetch 24 candidates: Cohere embeddings, cosine
   │                       distance over Chroma
   ├─ distance floor ..... nothing past 0.62 reaches the reranker or the model
   ├─ rerank ............. Cohere rerank-v3.5 reorders what cleared the floor
   ├─ diversify .......... max 2 chunks per section, keep the best 6
   └─ grounded prompt .... numbered sources, inline [n] citations, or a refusal
```

<details>
<summary><b>Why a distance floor, and why it is not enough</b></summary>

<br>

Vector search always returns *k* results, however bad the match. Scores are
**cosine distances, so lower is better** — `0.62` is not "62% similar", it is a
similarity of `0.38`. The bands are specific to the embedding model; measured
on this corpus with Cohere `embed-v4.0`:

| Band | Distance | Meaning |
| --- | --- | --- |
| Answering passage | 0.25 – 0.60 | The passage that answers an eval question |
| Adjacent but unanswerable | 0.65 | Basel III — banking, but not in the corpus |
| Off-topic | 0.72 – 0.86 | Cake recipes, football, prompt injection |

The floor sits at 0.62, in a gap only ~0.05 wide on each side — and that is the
whole problem: **a threshold cannot separate "the corpus covers this" from "the
corpus covers something that looks like this."** The hardest answerable case
(the DPDP Act's maximum penalty, 0.600) and the nearest unanswerable one (Basel
III, 0.654) are closer to each other than either is to its own band's centre.
The prompt rule is the second refusal gate for exactly that reason.

</details>

<details>
<summary><b>Refusal has three exits</b></summary>

<br>

1. **Nothing clears the floor** — a fixed refusal is returned and **no model call
   is made**. An out-of-corpus question costs zero LLM calls.
2. **Passages clear the floor but do not answer** — the prompt reserves the exact
   sentence *"This is out of my knowledge."* for this case.
3. **Partly answerable** — the covered part is answered with citations, then the
   uncovered part is named explicitly.

Refusal is detected by position, not by counting citations. A refusal routinely
cites while *describing* what it searched — "the sources cover digital lending
[1]" — which is the opposite of drawing on them. Counting those markers labelled
a refusal "2 sources" and invited the reader to conclude from them exactly what
the refusal had just said could not be concluded.

An early prompt enumerated the cases as labels — `(a) FULLY ANSWERED` — and the
model **echoed the labels into its answers**. Response shape has to be written as
behaviour, not as headings the model can copy.

</details>

## Guardrails

| Guardrail | Enforces |
| --- | --- |
| Distance floor | Off-topic never reaches the model |
| Grounded prompt | Answer only from the supplied sources |
| Citation range check | Every `[n]` points at a real source |
| Input cap | Questions over 2,000 chars are **rejected, not truncated** |
| Condense hardening | A follow-up is data, not instructions |
| Generation cap | Bounds per-turn cost |

A truncated question is a different question, and answering it confidently is
worse than declining.

**What these do not cover** — stated plainly because it matters:

- **Injection is stopped by the distance floor, not by injection detection.**
  `"Ignore all previous instructions..."` scores 0.80 and never reaches the model
  because it is semantically off-corpus, not because anything recognises it as an
  attack. An injection phrased in dense compliance language would clear retrieval,
  leaving only the prompt's "treat input as data" rule — and a prompt is a
  request, not a constraint.
- **Grounding itself is unmeasured.** The citation check confirms `[n]` is in
  range. It never asks whether passage *n* supports the sentence attached to it.
- **Chat history is global.** Conversations are not scoped per user, so any
  browser session sees every other session's questions. Fine for a local
  single-operator tool; a real gap for a shared deployment.
- **The whole conversation goes to the model, by choice.** `condense` uses
  only the last 4 messages to rewrite a follow-up for retrieval, but the answer
  step gets every prior turn so the model never loses what was said. The cost:
  a very long chat grows the prompt each turn. Earlier answers have their `[n]`
  markers stripped first, so the only source numbers in the prompt are the
  current turn's — otherwise an old `[2]` reads as this turn's `[2]`.

## Retrieval quality

`python eval_retrieval.py` — 30 cases, no LLM calls, about a minute, ~56 Cohere
calls (embed + rerank). Each case names a phrase the corpus really contains,
verified against the corpus when written rather than inferred from what
retrieval happened to return. `--no-rerank` runs the same cases on vector order
alone.

What the harness prints today:

```
answerable   26/26  (100% of questions retrieved the passage that answers them)
refused      4/4    (100% of out-of-corpus questions correctly returned nothing)
mean rank    1.3    (where the answering passage landed, 1 is best)

Hit@1 88%  Hit@3 96%  Hit@6 100%  MRR 0.924  (over all 26 answerable cases, a miss scores 0)
```

`mean rank` averages only over cases that passed, so it improves as cases fail —
the Hit@k and MRR line is the honest one, because a miss counts as 0 there. Over the same 30 cases, against the
August baseline (BGE embeddings over the pre-chunked corpus):

| Metric | Aug 2026 · BGE | Now · vector only | **Now · + Cohere rerank** |
| --- | --- | --- | --- |
| Hit@1 | 77% | 77% | **88%** |
| Hit@3 | 85% | 92% | **96%** |
| Hit@6 | 92% | 100% | **100%** |
| MRR (all cases) | 0.819 | 0.854 | **0.924** |
| Refused correctly | 4/4 | 4/4 | 4/4 |

Read the middle column with care: the embedding model (BGE → Cohere) and the
chunking (pre-chunked JSON → `pdf_to_chromadb.py`) changed together, so the vector-only
gain cannot be credited to either one alone. What it does show is that both
August failures — passages that were in the candidate list but discarded in
selection (the GDPR fine at candidate rank 8, the Companies Act's §166 at 16,
its title lost to body text) — now reach the top 6 on vector order alone, and
rank 1 with the reranker. **The reranker's effect is clean**, since it is the
only difference between the last two columns: same index, same cases, and 3
more answers moved to rank 1 (Hit@1 77% → 88%, MRR 0.854 → 0.924). The case it
leaves furthest down is the SEBI Security Operations Centre question (rank 5),
from a scanned document read by OCR.

> 30 cases is a small set; one case is ~4 points of Hit@1. Treat single-case
> differences as noise, and add cases before trusting a smaller change.

## Quickstart

Requires **Python 3.11**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
brew install tesseract           # OCR for the scanned PDFs (macOS)

cp .env.example .env             # set HF_TOKEN + COHERE_API_KEY; Langfuse optional
python pdf_to_chromadb.py build --reset # PDFs in source_pdfs/ → data/ JSON → chroma_db/
streamlit run app.py
```

Neither the PDFs nor the index are in the repo. Put the PDFs in
`source_pdfs/<FRAMEWORK_ID>/`, or set `CHROMA_INDEX_REPO` to a published index
and it is fetched on first use.

Or open the repo in **GitHub Codespaces / a VS Code dev container**:
[`.devcontainer/`](.devcontainer/devcontainer.json) installs
`requirements-dev.txt` and starts the app on port 8501. It still needs `.env`
and an index.

> **A full build is ~45 Cohere calls** (96 chunks each) and fits a free trial
> key, which allows 1,000 calls a month across embedding and reranking. Each
> question in the app costs two more, so a trial key serves roughly 450
> questions a month. `pdf_to_chromadb.py chunk` alone takes a few minutes and costs
> nothing; `pdf_to_chromadb.py embed --resume` continues an interrupted build;
> `--frameworks X --replace` re-does one framework. Changing
> `EMBED_MODEL` or `EMBED_DIM` invalidates every stored vector, and the app
> refuses to start against a mismatched index rather than answer from noise.

### Configuration

Everything is read from `.env` locally, or from Streamlit secrets when
deployed. [`.env.example`](.env.example) documents every variable; the ones
that matter:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `HF_TOKEN` | yes | — | Hugging Face token for the LLM. Use this name: `HUGGINGFACEHUB_API_TOKEN` alone is not enough |
| `COHERE_API_KEY` | yes | — | Embeddings and reranking (`CO_API_KEY` also accepted) |
| `HF_MODEL` | no | `Qwen/Qwen2.5-7B-Instruct` | Generation model |
| `CHROMA_INDEX_REPO` | deployed only | — | HF dataset holding a prebuilt index |
| `EMBED_MODEL` / `EMBED_DIM` | no | `embed-v4.0` / `1536` | Must match the index |
| `RERANK_MODEL` / `RAG_RERANK` | no | `rerank-v3.5` / `true` | Reranker, and an off switch for A/B runs |
| `RAG_MAX_DISTANCE` | no | `0.62` | Distance floor (the refusal gate) |
| `RAG_K` / `RAG_FETCH_K` / `RAG_MAX_PER_SECTION` | no | `6` / `24` / `2` | Sources kept, candidates fetched, per-section cap |
| `RAG_MAX_INPUT_CHARS` / `LLM_MAX_NEW_TOKENS` | no | `2000` / `1024` | Input and output caps |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | no | off | Langfuse tracing |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | no | off | LangSmith tracing |
| `RAG_DEBUG` | no | `false` | Show the nearest distance under a refusal |

### CLI

```bash
# corpus and index
python pdf_to_chromadb.py build --reset                # chunk + embed everything (asks to confirm the reset)
python pdf_to_chromadb.py chunk                        # PDFs → data/ JSON (no API calls)
python pdf_to_chromadb.py ocr                          # OCR the scanned PDFs into the cache only
python pdf_to_chromadb.py embed --resume               # data/ JSON → chroma_db/, continuing a stopped build
python pdf_to_chromadb.py embed --frameworks X --replace  # re-do one framework
python pdf_to_chromadb.py stats                        # what is indexed, and whether it is complete
python pdf_to_chromadb.py query "KYC periodic update" --framework RBI_KYC_MASTER_DIRECTIONS --k 5
python pdf_to_chromadb.py push-index --public <user>/<repo>  # publish the index for deployment

# evaluation and tests
python eval_retrieval.py --verbose              # retrieval quality (--no-rerank to compare, --strict for CI)
pytest                                          # 68 tests, ~1s, no network

# chat history
python store.py list                            # saved conversations
python store.py delete <id-prefix>              # delete one
python store.py clear                           # delete all, then vacuum
```

## Layout

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Streamlit UI — chat, history sidebar, framework scoping, citations |
| [`rag.py`](rag.py) | Retrieval policy, guardrails, prompt construction |
| [`pdf_to_chromadb.py`](pdf_to_chromadb.py) | PDFs → chunks → JSON → embeddings → Chroma, plus `stats` / `query` / `push-index` |
| [`vectorstore.py`](vectorstore.py) | Chroma + Cohere embeddings; the single source of truth for how text is embedded |
| [`observability.py`](observability.py) | Langfuse + LangSmith tracing, and 👍/👎 feedback |
| [`chat.py`](chat.py) | Model construction and per-turn run configuration |
| [`store.py`](store.py) | SQLite chat history |
| [`eval_retrieval.py`](eval_retrieval.py) | Retrieval evaluation harness |
| [`_bootstrap.py`](_bootstrap.py) | Copies Streamlit secrets into the environment; swaps in a newer sqlite for Chroma when the host's is too old |
| [`evals/retrieval.json`](evals/retrieval.json) | The 30 evaluation cases |
| [`tests/`](tests/) | Unit tests: grounding rules, chunking, embeddings, eval metrics |
| [`compliance-corpus-json/`](compliance-corpus-json/) | The generated corpus (reached as `data/`); per-framework `metadata.json` is the only hand-written input |
| [`.streamlit/`](.streamlit/) | Server config, and the secrets template for deployment |
| [`.devcontainer/`](.devcontainer/) | Codespaces / VS Code dev container |
| `requirements.txt` / `requirements-dev.txt` | Runtime only (what the deployed app installs) / plus PDF tooling, tests and eval extras |

`rag.py`, `vectorstore.py` and `store.py` deliberately import no Streamlit, so the
retrieval policy stays testable from the CLI and reusable behind an API. The
JSON corpus is generated output — fixes go into `pdf_to_chromadb.py`, never into the
JSON — so every build is reproducible from the PDFs.

**Nothing runs a model locally.** Embeddings and reranking (Cohere) and
generation (Hugging Face Inference API) are all API calls, so the deployed app
installs no torch. The cost is a monthly call budget on the free Cohere key.

## Observability

Every chat turn is one trace, in **Langfuse** and in **LangSmith** at once —
each switches on independently when its keys are set:

```
compliance-chat-turn     question → reply; session/thread = conversation id
  retrieve               candidates, cleared floor, best distance, hits
    rerank               model, relevance score per passage
  ChatHuggingFace        prompt, completion, tokens, latency
score: user-feedback     👍/👎 under each answer (Langfuse)
```

The turn records its outcome — answered, refused, or nothing found — and any
citation that pointed at a source that was never supplied, so both are
filterable.

## Documentation

| | |
| --- | --- |
| [`DEPLOY.md`](DEPLOY.md) | Deploying to Streamlit Community Cloud, and the traps in it |
| [`PHASES.md`](PHASES.md) | Why the environment is pinned as it is, plus the full engineering log |
| [`compliance-corpus-json/README.md`](compliance-corpus-json/README.md) | Corpus file layout and chunk fields |

## Limitations

- **Not legal advice.** It reports what the indexed text says. Regulation changes;
  the corpus is a snapshot.
- **The corpus is the boundary.** Anything outside those 25 frameworks is refused
  by design, including questions a general-purpose model could answer.
- **Six documents are OCR'd scans** (every SEBI document, the Consumer
  Protection Rules). Tesseract is good on clean print but not perfect, and OCR
  text has no font sizes, so footnotes and margin notes are not separated
  there the way they are in born-digital PDFs.
- **Heading detection is heuristic.** It is tuned on these 27 documents; a new
  framework with an unusual layout may chunk by page instead of by section
  (`qa_report.json` shows which style each document got).
- **Chat history is not durable when deployed.** Streamlit Cloud containers are
  ephemeral, so conversations are lost on restart. Local runs persist normally.
