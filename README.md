<div align="center">

# ⚖️ Compliance Chat

**A retrieval-augmented assistant over Indian and international regulation that
cites every claim — or refuses.**

[![Live app](https://img.shields.io/badge/demo-live-success?logo=streamlit&logoColor=white)](https://rag-based-compliance-chatbot.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](.python-version)
[![Frameworks](https://img.shields.io/badge/frameworks-25-informational)](#whats-indexed)
[![Retrieval](https://img.shields.io/badge/hit%406-92%25-brightgreen)](#retrieval-quality)
[![Tests](https://img.shields.io/badge/tests-39-brightgreen)](tests/)

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
prompt. [`corpus_repair.py`](corpus_repair.py) now repairs both problems at
ingest time — structurally, never keyed to a specific document — and fires on 27
chunks across 3 frameworks. The reflowed penalty table now comes back at rank 3.

## What's indexed

**27 documents · 25 frameworks · 2,602 source chunks → 4,060 indexed records**

| Sector | Frameworks |
| --- | --- |
| **Financial** | RBI (KYC, Cybersecurity, Digital Lending, Account Aggregator, IT Outsourcing) · SEBI (Cybersecurity, AML/CFT, Investment Adviser, Algo Trading) · IRDAI (Cyber Security, AML/CFT, Web Aggregator) · PCI-DSS v4 · PMLA |
| **Universal** | DPDP Act 2023 · DPDP Rules 2025 · IT Act 2000 · IT Rules 2011 · CERT-In Directions 2022 |
| **Corporate** | Companies Act 2013 · Labour Codes 2020 · Consumer Protection (E-Commerce) Rules 2020 · Telemedicine Guidelines 2020 |
| **Global** | GDPR · ISO 27001:2022 |

> **Half the corpus was nearly unsearchable.** Measured with the embedder's own
> tokenizer, **1,271 of 2,602 chunks (48.8%) exceed bge-base's 512-token limit**,
> the largest at 2,724. Indexed as-is, everything past token 512 is *silently*
> dropped — no error, no warning, just unretrievable text. Oversized chunks are
> split on the body only, with the locating header re-applied to each part and a
> `parent_chunk_id` back to one citable section.

## How it works

```
question
   │
   ├─ condense ........... rewrites a follow-up into a standalone question, so
   │                       "what about GDPR?" retrieves on more than four words
   ├─ input guard ........ length cap; the question is data, never instructions
   ├─ framework detect ... "the DPDP Act" scopes to the Act, bare "DPDP" scopes
   │                       to neither, two frameworks named means a comparison
   ├─ retrieve (k=24) .... BGE embeddings, cosine distance over Chroma
   ├─ distance floor ..... nothing past 0.45 reaches the model
   ├─ diversify .......... max 2 chunks per section
   └─ grounded prompt .... numbered sources, inline [n] citations, or a refusal
```

<details>
<summary><b>Why a distance floor, and why it is not enough</b></summary>

<br>

Vector search always returns *k* results, however bad the match. Scores are
**cosine distances, so lower is better** — `0.45` is not "45% similar", it is a
similarity of `0.55`. Measured bands on this corpus:

| Band | Distance | Meaning |
| --- | --- | --- |
| Genuine hit | 0.17 – 0.31 | The passage answers the question |
| Adjacent but unanswerable | 0.31 – 0.39 | Right topic, wrong document |
| Off-topic | 0.50 – 0.61 | Nothing relevant in the corpus |

The gap between the second and third band is narrow, and that is the whole
problem: **a threshold cannot separate "the corpus covers this" from "the corpus
covers something that looks like this."** Basel III, FEMA and GST questions all
retrieve six passages comfortably under the floor. Only the prompt rule refuses
them.

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
  `"Ignore all previous instructions..."` scores 0.52 and never reaches the model
  because it is semantically off-corpus, not because anything recognises it as an
  attack. An injection phrased in dense compliance language would clear retrieval,
  leaving only the prompt's "treat input as data" rule — and a prompt is a
  request, not a constraint.
- **Grounding itself is unmeasured.** The citation check confirms `[n]` is in
  range. It never asks whether passage *n* supports the sentence attached to it.
- **Chat history is global.** Conversations are not scoped per user, so any
  browser session sees every other session's questions. Fine for a local
  single-operator tool; a real gap for a shared deployment.

## Retrieval quality

`python eval_retrieval.py` — 30 cases, **no model calls**, no cost, under a
minute. Each case names a phrase the corpus really contains, verified against the
corpus when written rather than inferred from what retrieval happened to return.

What the harness prints today:

```
answerable   24/26  (92% of questions retrieved the passage that answers them)
refused      4/4    (100% of out-of-corpus questions correctly returned nothing)
mean rank    1.4    (where the answering passage landed, 1 is best)
```

**That `mean rank` flatters itself**, and it is worth knowing why. It averages
only over cases that *passed*, so the two failures are excluded from the number
meant to describe ranking quality — it improves as more cases fail. Measured
separately over the same 30 cases:

| Metric | Value | |
| --- | --- | --- |
| Hit@1 | 77% | |
| Hit@3 | 85% | |
| Hit@6 | 92% | what the harness reports today |
| MRR (all cases) | **0.819** | misses count as 0 |
| MRR (found only) | 0.887 | biased — ignores the 2 misses |

MRR over all cases is the honest headline: it carries hit rate and rank in one
number and cannot be improved by dropping hard cases. The Hit@1 → Hit@6 spread
(77% → 92%) is precisely the headroom a reranker would attack.

> One run in several reports 23/26 where three consecutive runs report 24/26 —
> MPS float variance flipping a borderline case. Treat a one-case difference as
> noise.

**Both known failures are ranking, not recall.** Neither passage is missing;
both clear the floor and sit inside the candidate list, then get discarded during
selection:

| Question | Gold passage | Rank | Distance |
| --- | --- | --- | --- |
| Maximum administrative fine under GDPR | `20 000 000` | 8 | 0.308 |
| A director's duties under the Companies Act | §166 `act in good faith` | 16 | 0.287 |

That reframes the fix — a cross-encoder rerank over the 24 candidates attacks it
more directly than more recall would.

## Quickstart

Requires **Python 3.11**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # add your Hugging Face token
python ingest.py build        # ~35 minutes, builds chroma_db/
streamlit run app.py
```

The index is **not** in the repo — it is ~99MB of generated artefact. Build it
locally, or set `CHROMA_INDEX_REPO` to a published copy and it is fetched on
first use (~9s).

> **A full rebuild costs ~35 minutes** and is not CPU-bound, so the machine looks
> idle throughout. Prefer `--frameworks X --replace` for targeted work (~4 min).
> Changing `EMBED_MODEL` silently invalidates every stored vector — it is not a
> configuration tweak.

### CLI

```bash
python ingest.py stats                        # what is indexed, and whether it is complete
python ingest.py query "KYC periodic update"  # search the index directly
python eval_retrieval.py --verbose            # retrieval quality
python store.py list                          # saved conversations
pytest                                        # 39 tests, ~0.7s
```

## Layout

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Streamlit UI — chat, history sidebar, framework scoping, citations |
| [`rag.py`](rag.py) | Retrieval policy, guardrails, prompt construction |
| [`vectorstore.py`](vectorstore.py) | Chroma + BGE; the single source of truth for how documents are embedded |
| [`chat.py`](chat.py) | Model construction and LangSmith run configuration |
| [`store.py`](store.py) | SQLite chat history |
| [`ingest.py`](ingest.py) | Corpus → index, plus `stats` / `query` / `push-index` |
| [`corpus_repair.py`](corpus_repair.py) | Structural fixes applied to source chunks at ingest time |
| [`eval_retrieval.py`](eval_retrieval.py) | Retrieval evaluation harness |

`rag.py`, `vectorstore.py` and `store.py` deliberately import no Streamlit, so the
retrieval policy stays testable from the CLI and reusable behind an API. The
corpus is treated as read-only input — repairs happen at ingest time, never by
editing the JSON, so every build is reproducible from pristine data.

**Embeddings run locally, generation does not.** BGE embeds 2,602 chunks and every
query in-process; routing that through a rate-limited API would make ingest slow
and retrieval scores non-reproducible between runs. The LLM is served over the
Hugging Face Inference API.

## Documentation

| | |
| --- | --- |
| [`DEPLOY.md`](DEPLOY.md) | Deploying to Streamlit Community Cloud, and the traps in it |
| [`PHASES.md`](PHASES.md) | Why the environment is pinned as it is, plus the full engineering log |

## Limitations

- **Not legal advice.** It reports what the indexed text says. Regulation changes;
  the corpus is a snapshot.
- **The corpus is the boundary.** Anything outside those 25 frameworks is refused
  by design, including questions a general-purpose model could answer.
- **30% of chunks carry a garbled `section_title`** — 781 of 2,602, body text
  spilled into the title field by the extraction pipeline. It lands in the
  embedding and in the reference line the reader sees. This is the direct cause of
  the Companies Act eval failure above.
- **Chat history is not durable when deployed.** Streamlit Cloud containers are
  ephemeral, so conversations are lost on restart. Local runs persist normally.
