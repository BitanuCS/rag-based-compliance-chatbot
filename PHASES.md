# Setup notes

Why the local environment is pinned the way it is, and the traps worth knowing
before changing it. [`README.md`](README.md) covers *how* to run the project;
this file covers *why* the dependency choices look like they do.

Environment bootstrapped 18 Aug 2026. Dependency pins re-verified 23 Aug 2026.

---

## Environment

| | |
| --- | --- |
| Python | 3.11 (pinned in `.python-version`) |
| Virtualenv | `.venv/` at the project root |
| Local device | Apple Silicon — `torch` uses MPS |
| Deployed device | Streamlit Community Cloud — CPU-only `torch` |

The local and deployed `torch` builds differ on purpose: `requirements.txt`
pulls `torch==2.6.0` from the PyTorch CPU index because the default PyPI wheel
ships CUDA and does not fit in a Streamlit container. Local dev runs 2.13.0
(MPS). There is no Linux cp311 CPU wheel at that version, so the deployed app
is deliberately one version behind.

---

## The LangChain version trap

**This is the one that will bite you.** `requirements.txt` pins:

```
langchain-core==0.3.74
langchain-huggingface==0.3.1
huggingface_hub==0.36.2
```

A plain `pip install langchain_huggingface` resolves to **1.2.2**, which
requires `langchain-core>=1.2.31`. That is a major-version jump, and every
other LangChain package in the 0.3.x line caps at `langchain-core<1.0.0`.
Installing it produces:

```
langchain-text-splitters 0.3.9 requires langchain-core<1.0.0,>=0.3.72,
  but you have langchain-core 1.5.6 which is incompatible.
```

`0.3.1` is the last release on the 0.x line and is the correct pin for this
stack. If you need `langchain-huggingface` 1.x, the whole LangChain stack has
to move to 1.x together — it is not a one-package upgrade.

Always install with an explicit version:

```bash
.venv/bin/python -m pip install "langchain-huggingface==0.3.1"
```

---

## Use the venv's Python, always

`pip install ...` in a fresh terminal hits the **system** Python at
`/Library/Frameworks/Python.framework/...`, not this project. The symptom is an
install that appears to succeed while the project keeps behaving as before —
and a system Python left with conflicting LangChain versions.

```bash
source .venv/bin/activate     # then plain python / pip are correct
# or address it explicitly:
.venv/bin/python -m pip install ...
.venv/bin/python ingest.py stats
```

After any install, confirm the environment is still coherent:

```bash
.venv/bin/python -m pip check      # expect: No broken requirements found.
```

If the system Python does get into a broken state, this restores it:

```bash
python3 -m pip install "langchain-core>=0.3.72,<1.0.0"
```

---

## Secrets

`.env` is gitignored and holds the Hugging Face token. Both names work —
`HUGGINGFACEHUB_API_TOKEN` is what LangChain's `HuggingFaceEndpoint` looks for,
`HF_TOKEN` is the `huggingface_hub` convention:

```
HUGGINGFACEHUB_API_TOKEN=hf_...
```

The token needs the **"Make calls to Inference Providers"** permission. A
classic read token also works. `.env.example` documents every variable the app
reads; copy it, never commit the filled-in version.

Before any push, confirm the token is not staged:

```bash
git status --porcelain | grep -E "\.env$"     # no output = safe
```

For deployment, the same values go in Streamlit's secrets UI rather than a
file — see [`DEPLOY.md`](DEPLOY.md). `.streamlit/secrets.toml` is gitignored for
the same reason `.env` is.

---

## `data/` is a symlink

`data → compliance-corpus-json` — the parsed corpus is committed under its real
name and reached through the link. Git stores the symlink itself, so a clone on
macOS or Linux resolves it correctly and nothing is duplicated in the repo.

The corpus is treated as read-only input. Fixes to source chunks are applied at
ingest time in [`corpus_repair.py`](corpus_repair.py), never by editing the
JSON in place, so re-running ingest is always reproducible from pristine data.

---

## Model choice

The LLM is served over the **Hugging Face Inference API**, not run locally —
`chat.py` builds a `ChatHuggingFace` around a `HuggingFaceEndpoint`
(`Qwen/Qwen2.5-7B-Instruct` by default).

Embeddings are the opposite: BGE runs **locally** in
[`vectorstore.py`](vectorstore.py). Routing 2,602 chunks plus every query
embedding through a rate-limited API would make ingest slow and retrieval
scores non-reproducible between runs.

`chat.py` has no chat loop of its own, deliberately. An ungrounded assistant
answering compliance questions from model memory is the failure mode this
project exists to prevent, so the only path to the model runs through
`rag.build_prompt`. To inspect the corpus directly, use
`python ingest.py query "..."` instead.

---

## Chat history is SQLite, and the id is shared with LangSmith

[`store.py`](store.py) persists the UI's chat history to `chat_history.db`
(gitignored) in two tables — `conversations` and `messages`. It imports neither
Streamlit nor LangChain, so it stays usable from a CLI or a future API layer;
messages are stored as plain `("user"/"assistant", text)` rows and mapped to
LangChain objects by the caller.

Three decisions that are not obvious from reading it:

**The conversation id *is* the LangSmith thread id.** `app.py` mints a UUID
when a chat starts — before the first message — and passes it to both
`chat.run_config(thread_id)` and `store.create_conversation(..., conv_id=...)`.
Minting it at save time instead would leave the first turn traced under a
different id than every later turn. Because the two agree, a conversation id
from the sidebar pastes straight into LangSmith's thread filter.

**A conversation row is only written after a turn completes.** A new chat holds
a thread id but no database row until the model actually replies, so abandoned
chats and failed requests leave nothing behind. This is why a fresh chat shows
no sidebar entry until its first answer arrives.

**Every call opens its own connection and closes it.** Streamlit reruns the
script across threads, so a shared connection would need locking for no real
gain. Note that `with sqlite3.connect(...) as conn` **commits but does not
close** — relying on it alone leaks a file descriptor per rerun, which in a long
session eventually hits the open-file limit. `_session()` does both.

### Deleting looks broken when it is not

Deleted rows are really gone, but two things make it look otherwise:

- **The file never shrinks.** SQLite moves freed pages onto a reuse list rather
  than truncating, so `chat_history.db` holds its size no matter how much you
  delete. `VACUUM` is what shrinks it, and it is purely cosmetic.
- **GUI viewers cache.** DB Browser for SQLite and the VS Code SQLite extension
  hold their own connection and show a snapshot from before the app's write.
  Reconnect before concluding anything.

`store.py` runs as a CLI, and is the reliable check because it reads fresh from
disk:

```bash
.venv/bin/python store.py list              # what is actually on disk right now
.venv/bin/python store.py delete <id-prefix>
.venv/bin/python store.py clear             # wipe everything, then vacuum
.venv/bin/python store.py vacuum
```

`list` prints row counts plus `orphan_messages` — messages whose conversation is
gone. Any number above zero there is a genuine bug; zero means deletes are clean.

---

## LangSmith tracing

Off unless the environment says otherwise. Both variable prefixes are read, the
older `LANGCHAIN_*` first-party names and the current `LANGSMITH_*` ones, so
either works:

```
LANGCHAIN_TRACING_V2=true      # or LANGSMITH_TRACING=true
LANGCHAIN_API_KEY=lsv2_pt_...  # or LANGSMITH_API_KEY
LANGCHAIN_PROJECT=compliance-rag
```

`load_dotenv()` runs at import, so **a running Streamlit server keeps the old
value** — restart it after flipping the flag.

`chat.run_config()` sets `session_id`, `thread_id` and `conversation_id` in the
run metadata to the same value. LangSmith groups runs into a thread by
`session_id`, with the other two as accepted aliases; setting all three keeps
grouping working regardless of which key the UI version keys off. Each turn also
carries `model` and `turn` metadata, and runs are tagged `compliance-chat`.

Traces only appear once a model call actually happens. If the Hugging Face
endpoint is unreachable — a DNS failure resolving `huggingface.co` looks like
`NameResolutionError` in the UI's error banner — the call fails before reaching
the model, so LangSmith shows error runs or nothing at all. Rule out the network
before concluding tracing is misconfigured.

---

## The corpus is already chunked — split only what overruns

`data/all_chunks.json` ships 2,602 retrieval-ready chunks, each carrying two text
fields that are **not** interchangeable:

| Field | Use |
| --- | --- |
| `embed_text` | Embed this. `text` with `[framework \| regulator \| date]` and the breadcrumb prefixed |
| `text` | Display this. Clean clause text |

The header is the point: it encodes *where* a clause sits, not only what it says.
Re-cutting the corpus with a plain text splitter would throw that away, so
[`ingest.py`](ingest.py) does not re-chunk. It splits only what it must.

What it must split is substantial. Measured with the embedder's own tokenizer —
not the corpus's `token_estimate`, which is `char_count/4` and drifts — **1,271 of
2,602 chunks (48.8%) exceed bge-base's 512-token limit**, the largest at 2,724.
Indexed as-is, everything past token 512 is silently dropped and becomes
unretrievable. Roughly half the corpus would have been quietly unsearchable.

So oversized chunks are split on the **body only**, with the header re-applied to
each part, and every part keeps a `parent_chunk_id` back to one citable section.
Ids are deterministic (`chunk_id`, or `chunk_id#pN`), so re-running ingest upserts
in place rather than duplicating.

Two invariants worth knowing before touching this:

- `embed_text` ends with `text` for all 2,602 chunks, which is how the header is
  recovered by slicing. `header_of()` raises if that ever stops holding, rather
  than indexing a mangled header.
- Chroma stores `str | int | float | bool` only. This corpus breaks that twice:
  `chapter` is `None` on most chunks, and `section_path` is a list. Both are
  normalised in `clean_metadata()`; a new corpus field of another type will fail
  at insert.

Display text is recovered by slicing the header back off using `header_chars`,
rather than storing all 13 MB of clause text a second time.

---

## Scores are distances, not similarities

The collection is created with `hnsw:space: cosine` and vectors are L2-normalised
at embed time. **`hnsw:space` is frozen when the collection is first created** —
Chroma's default is `l2`, and changing it later means a full rebuild.

Every number surfaced in the UI and the CLI is a **cosine distance, where lower is
better**:

```
distance = 1 − cosine_similarity
```

Verified against a hand-computed dot product: Chroma's `0.244050` versus
`1 − 0.755950`, matching to six decimals. So `0.45` is not "45% similar" — it is a
similarity of `0.55`.

Measured bands on this corpus:

| Band | Distance | Meaning |
| --- | --- | --- |
| Genuine hit | 0.17 – 0.31 | The passage answers the question |
| Adjacent but unanswerable | 0.31 – 0.39 | Right topic, wrong document |
| Off-topic | 0.50 – 0.61 | Nothing relevant in the corpus |

The gap between the second and third band is narrow, and that is the whole
problem: **a distance threshold cannot separate "the corpus covers this" from
"the corpus covers something that looks like this."** Basel III, FEMA and GST
questions all retrieve six passages comfortably under the floor. Only the prompt
rule refuses them.

---

## The refusal policy has three exits

Retrieval alone returns `k` results however bad the match, so refusing is layered:

1. **Nothing clears the distance floor** (`RAG_MAX_DISTANCE`, default `0.45`) —
   `rag.NO_CONTEXT` is returned directly and **no model call is made**.
2. **Passages clear the floor but do not answer** — the system prompt reserves the
   exact sentence `"This is out of my knowledge."` for this case.
3. **Partly answerable** — the covered part is answered with citations, then the
   uncovered part is named. The refusal sentence must *not* appear here.

That third case caused a real bug worth not reintroducing. An early version of the
prompt enumerated the cases as labels — `(a) FULLY ANSWERED`, `(b) PARTLY
ANSWERED` — and the model **echoed the labels into its answers**. Instructions
about response shape have to be written as behaviour, not as headings the model
can copy. The prompt now ends that rule with an explicit "never label your
response."

`rag.is_refusal()` detects case 2 by matching the reserved sentence, deliberately
**not** by counting citations: a refusal routinely cites while *describing* what it
searched — "the sources cover digital lending [1], cybersecurity [2]" — which is
the opposite of drawing on them. Counting those markers labelled a refusal
"2 sources" and invited the reader to conclude from them exactly what the refusal
had just said could not be concluded.

The reference list therefore renders **only passages the answer actually cites**.
A refusal cites nothing, so it renders no reference list at all.

---

## Guardrails, and what they do not cover

| Guardrail | Where | Enforces |
| --- | --- | --- |
| Distance floor | `rag.MAX_DISTANCE` | Off-topic never reaches the model |
| Grounded prompt | `rag.SYSTEM_PROMPT` | Answer only from supplied sources |
| Citation range check | `rag.validate_citations()` | Every `[n]` points at a real source |
| Input cap | `rag.MAX_INPUT_CHARS`, `chat_input(max_chars)` | Rejects >2,000 chars, both sides |
| Condense hardening | `rag.condense()` | Follow-up is data, not instructions |
| Generation cap | `chat.MAX_NEW_TOKENS` | Bounds per-turn cost |

Oversized input is **rejected, not truncated** — a truncated question is a
different question, and answering it confidently is worse than declining.

Three limits to be honest about:

- **Citation validation has a false-positive mode.** A model that *mentions* a
  number while refusing to use it — "the sources do not include [9]" — trips the
  check. The warning is redundant there, not wrong. Flagging conservatively is the
  safer default.
- **Injection is stopped by the distance floor, not by injection detection.**
  `"Ignore all previous instructions..."` scores 0.52 and never reaches the model —
  because it is semantically off-corpus, not because anything recognises it as an
  attack. An injection phrased in dense compliance language would clear retrieval,
  leaving only the prompt's "treat input as data" rule. That rule held under
  testing, but a prompt is a request, not a constraint.
- **`RAG_MAX_DISTANCE` is now doing double duty** as a relevance filter *and* a
  security control. Raising it loosens both.

---

## Rebuilding the index costs ~35 minutes

A full `ingest.py build --reset` re-embeds every record on MPS and took roughly 35
minutes wall clock. It is not CPU-bound, so the machine looks idle throughout —
that is expected, not a hang.

Consequences worth internalising:

- Changing `EMBED_MODEL` silently invalidates every stored vector and costs a full
  rebuild. It is not a configuration tweak.
- Prefer targeted work: `ingest.py build --frameworks X` upserts in place because
  ids are deterministic, and `--limit N` gives a fast smoke test.
- Run full rebuilds in the background from the start, not in a foreground call
  that will time out.

`ingest.py stats` is the check that matters after any build. It compares distinct
`parent_chunk_id`s per framework against `data/validation.json` and asserts that
**no stored record exceeds 512 tokens** — the truncation the split step exists to
prevent, verified rather than assumed.

---

## When answers stop and retrieval keeps working

Hugging Face Inference credits run out, and the failure is loud but easy to
misread:

```
402 Client Error: Payment Required for url: https://router.huggingface.co/...
You have depleted your monthly included credits.
```

Only *generation* is affected. Embeddings run locally, so retrieval, the distance
floor, refusals and `ingest.py query` all keep working — which makes it look like
a model bug rather than a billing one. Rule out the 402 before debugging the
prompt.

`requirements.txt` already carries `google-genai` and `groq`, and `.env.example`
documents `LLM_PROVIDER`, `GOOGLE_API_KEY` and `GROQ_API_KEY` as alternates behind
the same wrapper.

---

## Session log — 23 August 2026

Five problems, found by pulling on one wrong answer. The thread is worth keeping:
the answer looked fine, cited its sources, and was wrong for a reason that lived
three layers below the model.

### The wrong answer, and what caused it

Asked whether a Data Principal can be penalised under the DPDP Act, the system
replied that the Act *"does not specify a maximum penalty."* It does — up to
₹250 crore, in the Schedule.

The text was indexed the whole time. The extraction pipeline had not given the
Schedule its own section, so it sat inside the chunk labelled
`Chapter IX > 44 Section 44`, and the breadcrumb is part of what gets embedded.
A search for "maximum penalty" matched nothing, because the penalties were
filed under a heading about amendments to other Acts.

The same pipeline flattens tables column by column: every breach description
first, then every serial number, then every rupee amount. The breach → amount
mapping — the entire content of the table — was gone before it reached Chroma.

[`corpus_repair.py`](corpus_repair.py) repairs both at ingest time. It fired on
**27 chunks across 3 frameworks**, not just DPDP: the Companies Act's Schedule
VII (CSR activities) and Schedule III (financial statements) were also filed
under `Section 252`, and 18 IRDAI web-aggregator forms under `General`.

Both repairs are structural, never keyed to a document. A table is reflowed only
when all three columns are found **and their row counts match**; anything else
passes through untouched, because a repair firing on a shape it does not
understand corrupts text that was fine. One false positive surfaced during the
corpus-wide sweep and is now guarded against: the Companies Act contents page
lists `SCHEDULE I.`, `SCHEDULE II.` …, which would have filed the Act's own
enacting words under Schedule I. Bare headings are trusted only when one stands
alone.

### `--replace` exists because upsert is not enough

Deterministic ids mean a re-run upserts — but only over the ids the new run
happens to produce. A chunk that used to split into `#p1`/`#p2` and now fits in
one record leaves those parts behind as orphans, answering to text that no
longer exists. That is exactly what the schedule lift causes.

```bash
.venv/bin/python ingest.py build --frameworks DPDP_ACT_2023,COMPANIES_ACT_2013,IRDAI_WEB_AGGREGATOR_REGS_2017 --replace
```

`--replace` deletes the named frameworks' records first, making a targeted
re-ingest equivalent to a rebuild of that slice — ~4 minutes against the ~35 a
full `--reset` costs. It refuses to run without `--frameworks`.

`ingest.py stats` now reports `schedules lifted into their own records` and
excludes repaired records from the per-framework parity count, since they are
lifted out of a parent rather than being parents themselves. Post-repair it
reads 33 lifted records and *all frameworks fully indexed ✓*.

### Retrieval returned one section six times

On a KYC question, five of six retrieved passages were the same section, same
pages. The model thought it was reading six sources and was reading one.

`rag.diversify()` now pulls `RAG_FETCH_K` (24) candidates and caps any single
section at `RAG_MAX_PER_SECTION` (2). Two rather than one because an obligation
and its proviso are often split across consecutive parts. That question now
returns six distinct sections.

That alone did not fix DPDP: the question named a framework and retrieval still
returned five GDPR passages, because GDPR says more about penalties in more ways.
`rag.detect_framework()` scopes to a framework the question names — naming word
(`dpdp`) worth two, instrument word (`act`) worth one, so "the DPDP Act" picks
the Act over the Rules while bare "DPDP" picks neither. Tokens of three
characters or fewer must appear capitalised, or `IT_ACT_2000` matches the pronoun
in "is **it** mandatory". Two frameworks named means a comparison, so nothing is
scoped. A sidebar scope always overrides detection, and a detection that finds
nothing falls back to the whole corpus — a guess is never allowed to cause a
refusal.

### Tests

`tests/`, 39 cases, pure functions only, ~0.7s. `pytest==9.1.1` added to
`requirements.txt`.

```bash
.venv/bin/python -m pytest tests/ -q
```

They earn their place immediately: the suite caught that *"compare DPDP Act and
GDPR"* scoped to DPDP and would have hidden half the comparison. Frameworks are
now only rivals for one scope when the words that matched them overlap — "RBI"
and "RBI KYC" are one family, "DPDP" and "GDPR" are two.

Nothing here touches the index. A test that depends on 2,602 embedded chunks
tells you the corpus changed, not that the code broke; that job belongs to the
harness below.

### Measuring retrieval instead of eyeballing it

```bash
.venv/bin/python eval_retrieval.py [--verbose] [--strict]
```

30 cases in [`evals/retrieval.json`](evals/retrieval.json), **no model calls**, under
a minute. Each case names a phrase the corpus really contains — verified against
`data/all_chunks.json` when written, not guessed from what retrieval happened to
return. A case fails when no retrieved passage carries that phrase, meaning the
model could not have answered correctly however good it is.

```
answerable   24/26  (92%)
refused       4/4   (100%)
mean rank     1.4
```

Two known failures, both real, both lexical needles that embeddings rank poorly:

| Question | Wanted passage |
| --- | --- |
| Maximum administrative fine under GDPR | `20 000 000`, ranks 8th |
| A director's duties under the Companies Act | §166 `act in good faith`, ranks 16th |

Raising the section cap to 3 or 4 does not help — measured. The fix is hybrid
BM25 + vector retrieval; `rank-bm25` is already in `requirements.txt`. Left
undone deliberately: it needs its own measurement cycle, which now exists.

**Caveat on reading the number.** One run in several reported 23/26 where three
consecutive runs reported 24/26 — MPS float variance flipping a borderline case.
Treat a one-case difference as noise. The harness calls `os._exit()` rather than
returning normally, because chromadb and torch race each other through
interpreter teardown and abort, which would otherwise make every run look like a
failure to anything reading the exit code.

### Smaller things

- **The refusal sentence is position-anchored now.** `rag.is_refusal()` was a
  substring test, so a model that wrote seven cited paragraphs and then tacked
  *"This is out of my knowledge."* on the end was classed a refusal and had its
  citations suppressed — the answer visibly rested on sources while the panel
  showed none. It now matches only at the start, where the prompt puts a real
  refusal, and `rag.strip_stray_refusal()` removes the sentence when it trails a
  substantive answer. Both refusal exits render the identical `NO_CONTEXT` text,
  so a reader never has to interpret two wordings.
- **Retrieval runs before any model is touched.** An out-of-corpus question costs
  zero LLM calls; the condense step fires only when the raw question misses *and*
  there is history, where it can still turn a refusal into an answer. An
  out-of-corpus follow-up still costs one condense call — unavoidable, since a
  vague follow-up cannot be judged out of scope before resolving what it refers to.
- **References render under the answer**, grouped one line per distinct location
  carrying every marker that points at it (`[1][2][4]`), with the passage text
  collapsed behind them. `rag.group_references()` holds that logic so it is
  testable outside Streamlit.
- **Errors no longer print raw exception text** to the page — the detail goes to
  the server log, the reader gets the exception class name. Endpoint errors can
  carry request URLs, and a compliance officer can do nothing with a stack trace.
- **Sidebar**: the LangSmith project/thread footer is gone (tracing itself is
  untouched), and history rows clip to one line with the full title in the
  tooltip. The clipping CSS hooks Streamlit's internal `st-key-*` class, which
  needs ≥1.39 — if a future version renames it, rows revert to wrapping.

### Verified, and not

Checked: the repair sweep across all 2,602 chunks, the re-ingest, `ingest.py
stats` parity, 39 tests, the eval at four `k`/cap settings, and the original
failing question end to end through retrieval — the reflowed penalty table now
comes back at rank 3.

Not checked: the Streamlit UI itself. Every module compiles and the answer path
reads correctly, but no one has driven the app since these changes.

---

## Review session — 23 August 2026 (later the same day)

**Nothing in the tree was changed.** This was a read-and-measure pass over the
system as it stood after the session above. Everything below is a finding or a
recommendation; no code, corpus or index was touched. Where a number here
disagrees with one further up, this section is the later measurement.

### Both known eval failures are ranking, not recall

The table above calls them *"lexical needles that embeddings rank poorly"*, which
is right, but understates how close they already are. Measured directly against
the store:

| Question | Gold passage | Raw rank | Distance | Inside floor? |
| --- | --- | --- | --- | --- |
| Maximum administrative fine under GDPR | `20 000 000` | 8 | 0.308 | yes |
| A director's duties under the Companies Act | §166 `act in good faith` | 16 | 0.287 | yes |

Neither passage is missing. Both clear the 0.45 floor comfortably and both sit
inside `RAG_FETCH_K` (24), so the candidate list already contains the answer in
both cases — it is discarded during selection. That reframes the fix: the
bottleneck is **ranking**, and a cross-encoder rerank over the 24 candidates
attacks it more directly than more recall does. `bge-reranker-base` pairs with
the `bge-base` embedder already in use. Hybrid BM25 still helps, particularly for
the Companies Act case where the section is literally titled *Duties of
directors*, but it is no longer the only lever.

This also explains the note above that raising the section cap to 3 or 4 does not
help — confirmed here independently. For the GDPR question, GDPR's recitals hold
2 of the 6 final slots, and all 173 recitals share the single breadcrumb
`> 0 Preamble`, so the whole non-binding preamble is treated as one "section"
with the same standing as an operative article. Raising the cap lets the preamble
expand alongside Article 83, and the gold passage stays out either way. Worth
keeping in proportion: corpus-wide, recitals are only **9%** of returned context
and touch 10 of 26 answerable cases — this is a GDPR-shaped problem, not a
systemic one.

### 30% of chunks carry a garbled `section_title`

781 of 2,602 chunks have body text spilled into the title field by the extraction
pipeline:

```
'Incorporation of company.-(1) There shall be filed with the Registrar within who'
```

| Framework | Chunks |
| --- | --- |
| `COMPANIES_ACT_2013` | 184 |
| `IRDAI_AML_CFT_GUIDELINES` | 109 |
| `RBI_CYBERSECURITY_FRAMEWORK` | 96 |
| `PCI_DSS_V4` | 91 |
| `PMLA_AML_CFT` | 58 |

This is not cosmetic. `ingest.py` prefixes the breadcrumb into `embed_text`, so a
broken title lands in the embedding, in the `diversify()` section key, and in the
reference line the reader sees.

The Companies Act failure is the visible symptom. The chunk that answers it,
`COMPANIES_ACT_2013-0160-c01`, has this header:

```
Companies Act, 2013 > Chapter XI > 166 Section 166
```

The real title — *Duties of directors* — was lost, so a question about director
duties has nothing in the header to match against. The body then opens mid-clause
and runs straight into amendment apparatus (`1. The proviso ins. by Act 1 of
2018, s. 52...`). Recovering titles and stripping statutory footnotes is the same
class of structural repair `corpus_repair.py` already does, and it would need a
targeted `--replace` re-ingest of the affected frameworks.

### The eval harness reports a metric that flatters itself

`eval_retrieval.py` builds `mean_rank` from `found_at`, which is appended to only
when a case **passes**. The two failures are excluded from the number meant to
describe ranking quality, so it improves as more cases fail. Same bias in any
MRR computed that way. Measured over the existing 30 cases:

```
Hit@1            77%
Hit@3            85%
Hit@6            92%    <- what the harness reports today
MRR (all cases)  0.819  <- misses count as 0
MRR (found only) 0.887  <- biased, ignores the 2 misses
mean rank        1.42   <- current metric, also found-only
```

**MRR over all cases** is the better single headline: misses score 0, so one
number carries hit rate and rank together and cannot be improved by dropping hard
cases. The Hit@1 → Hit@6 spread (77% → 92%) is the reranker headroom, stated
precisely.

Two more worth tracking, both free and automatic:

- **Floor headroom.** The hardest answerable question bottoms out at 0.393; the
  easiest out-of-corpus question at 0.499 — a **+0.106 margin** around the 0.45
  floor, which sits near the middle of it. Nothing currently reports this, so the
  first sign of the floor misfiring as the corpus grows would be a failing case
  rather than a shrinking margin.
- **Recital contamination.** Whether a returned passage is a preamble chunk is
  readable straight off the breadcrumb; no labels needed.

Beyond retrieval, the layer with no measurement at all is grounding.
`rag.validate_citations()` checks that `[n]` is in range — necessary, nowhere
near sufficient. It never asks whether passage *n* supports the sentence attached
to it. The cheapest high-signal additions are **threshold exactness** (regex the
expected figure out of the generated answer — `20 000 000`, `72 hours`; the
`expect_text` field already does this for retrieval) and **framework attribution**
(did a DPDP question get answered out of GDPR), both automatic. Citation support
precision needs a judge model.

### Three bugs in the chat path

None of these are hypothetical; all three are visible in the current code.

- **Citation markers collide across turns.** `rag.build_prompt()` splices raw
  history in ahead of the new evidence block. Turn 1's stored answer says `[2]`
  meaning a DPDP passage; turn 2 supplies a fresh `[1]`–`[6]` for entirely
  different passages. The model reads both numbering schemes as one namespace.
  Historical markers need stripping or renumbering.
- **History is windowed everywhere except where it matters.** `rag.condense()`
  correctly takes `history[-4:]`, but `build_prompt` passes `*(history or [])` —
  every prior turn in full, alongside six sources. A long chat silently pushes
  the evidence out of the context window, which is the exact failure
  `MAX_INPUT_CHARS` exists to prevent.
- **`uncited` is computed and always discarded.** `validate_citations` returns it;
  `app.py` throws it away as `_, invalid, _`. Either use it to trim unused sources
  or drop it from the signature.

### Chat history is global

`store.list_conversations()` returns every conversation with no per-user scoping,
and the sidebar renders all of them. Any browser session sees every other user's
compliance questions. Fine for a single-operator local tool, not fine for anything
deployed — and `DEPLOY.md` now describes deploying it.

### Verified, and not

Checked, by running them: the 39-case test suite (passes, ~0.6s), the retrieval
eval (24/26, 4/4, matching the numbers above), the raw rank and distance of both
failing gold passages, the 781-chunk title count across all 2,602 chunks, the
recital share across all 26 answerable cases, and the floor margin.

Not checked: everything in *Three bugs in the chat path* is read off the source,
not reproduced against a live model — the reasoning is short enough to follow in
each case, but no one has driven a multi-turn chat to watch the markers collide.
The metrics were measured against a working tree in which `app.py`, `ingest.py`,
`vectorstore.py` and `requirements.txt` were concurrently modified by another
session doing deployment work; `rag.py` and the corpus were untouched, so the
retrieval numbers hold, but they were not taken against a clean tree.

---

## Superseded

The first pass at this project used a different layout — `config.py`,
`corpus.py`, `verify_env.py` and a `retrieval/` `generation/` `graph/` package
split, described in a `BUILD_PLAN.md` that is no longer in the tree. That
scaffold was replaced by the current flat module layout
(`ingest.py`, `vectorstore.py`, `rag.py`, `store.py`, `app.py`) documented in
the README's Layout table. Only the environment decisions above carried
forward. If you find a reference to those files anywhere, it is stale.
