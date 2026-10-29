# Interview prep — Compliance Chat

Checked against the code and the built index on 29 October 2026. This replaces
`interview prep.pdf`, which describes the **August** system (BGE, 0.45 floor,
2,602 chunks, 39 tests, no reranker). Do not quote numbers from the PDF.

---

## The 30-second pitch

> A RAG assistant over 25 Indian and international regulations — RBI, SEBI,
> IRDAI, DPDP, GDPR, PCI-DSS and others. Every claim cites a framework, section
> and page you can check, and when the corpus doesn't cover a question it
> refuses instead of improvising. The interesting work was upstream of the
> model: I built the pipeline from raw PDFs to the vector index myself, because
> the bug that shaped the project was a wrong answer caused by bad extraction,
> not by the LLM.

## The 3-minute walkthrough

1. **Problem.** A confident wrong answer about a regulatory obligation gets
   acted on, so an unsourced compliance assistant is worse than none. Design
   rule: cite, or refuse.
2. **The DPDP bug (your best story).** Asked for the DPDP Act's maximum
   penalty, an early version said the Act "does not specify one". It does —
   ₹250 crore, in the Schedule. Two root causes, both in extraction: the
   Schedule was filed under a heading about amending other Acts (and the heading
   is embedded, so "maximum penalty" matched nothing), and the penalty table was
   flattened column by column, so breach → amount was lost. The answer looked
   fine and cited sources. **Lesson: retrieval quality is the ceiling.**
3. **Fix: own the pipeline.** `pdf_to_chromadb.py` — PyMuPDF text, Tesseract OCR
   for 6 scanned PDFs, structure-aware chunking (one section = one chunk), tables
   rebuilt row by row from the `(1) (2) (3)` column-number row.
4. **Query path.** Input guard → framework detection → fetch 24 candidates
   (Cohere embeddings, Chroma, cosine) → distance floor 0.62 → Cohere rerank →
   max 2 per section, keep 6 → grounded prompt with `[n]` citations.
5. **Measured.** 30-case retrieval eval: 26/26 answerable, 4/4 refused, Hit@1
   77% → 88% from the reranker, MRR 0.854 → 0.924.
6. **Honest limits.** Grounding (does passage *n* support the sentence?) is not
   measured; chat history isn't per-user or durable when deployed.

---

## Fact sheet — current numbers

| Thing | Value |
| --- | --- |
| Corpus | 27 PDFs · 25 frameworks · 2,816 sections · **4,299 chunks** |
| Sectors | Financial, Universal (Indian cross-sector), Corporate, Global |
| OCR | 6 scanned PDFs (all SEBI, Consumer Protection Rules), Tesseract |
| Chunking | one per section; split past **512 tokens** (`cl100k_base`), **64-token overlap**, paragraph → line → sentence → clause |
| Chunk header | `[framework \| regulator \| date]` + `Framework > Chapter > Section` breadcrumb, embedded with the text |
| Embeddings | Cohere **`embed-v4.0`, 1536 dims**, `search_document` / `search_query` (asymmetric) |
| Vector store | Chroma, `hnsw:space=cosine`, collection `compliance`, ~121 MB |
| Reranker | Cohere **`rerank-v3.5`**, after the floor; falls back to vector order on failure |
| Retrieval | fetch **24** → floor **0.62** → rerank → max **2 per section** → keep **6** |
| LLM | Hugging Face Inference API; `HF_MODEL` = Llama-3.3-70B-Instruct in `.env.example`, default Qwen2.5-7B-Instruct |
| Input cap | 2,000 chars, **rejected, not truncated** |
| Generation cap | 1,024 new tokens |
| Condense | only after a miss with history; last **4 messages**; rewrite > 300 chars discarded |
| Observability | Langfuse **and** LangSmith, one trace per turn: retrieve > rerank > LLM; 👍/👎 scored in Langfuse |
| Storage | SQLite chat history (`store.py`) |
| UI / deploy | Streamlit, Community Cloud; index fetched from a HF dataset repo |
| Tests | **68**, ~1 s, no index or network |
| Eval | 30 cases (26 answerable, 4 refusal); Hit@1 88%, Hit@3 96%, Hit@6 100%, MRR 0.924 |
| Distance bands | answers 0.25–0.60 · Basel III 0.654 · off-topic 0.72–0.86 · injection 0.80 |
| Cost | free Cohere trial: 1,000 calls/month; build ≈ 45 calls; 2 calls per question (~450 questions/month) |

**Do not say:** BGE, 0.45, 2,602 chunks, 39, 61 or 65 tests, "k=24 sources",
LangGraph, "runs locally on MPS", "injection scores 0.52".

---

## Questions you should expect

### Basics

**Why RAG and not fine-tuning?** Regulation changes; re-indexing one framework
beats retraining. Fine-tuning gives tendencies, not citable recall. And refusing
needs to know at query time what the corpus covers.

**What does "grounded" mean here — what's enforced vs requested?** Enforced in
code: the distance floor (off-topic never reaches the model) and the citation
range check (every `[n]` must exist). Requested: "answer only from the sources"
is a prompt rule, and a prompt is a request, not a constraint.

**How does refusal work?** Three exits. (1) Nothing clears the floor → fixed
refusal, **zero LLM calls**. (2) Passages clear the floor but don't answer → the
model writes the reserved sentence "This is out of my knowledge." (3) Partly
answerable → answer the covered part with citations, name the gap. Refusal is
detected **by position** (the reply starts with it), not by counting citations,
because refusals cite while describing what they searched.

### Retrieval

**Why structure-aware chunking instead of fixed-size?** A regulation is already
chunked by its author: one section is one obligation, cited by number. Cutting
elsewhere splits a rule from its proviso, and the reader loses "Section 8(5)".

**Why embed the breadcrumb header?** A clause is only findable by its topic if
the embedding knows where it sits. That's also exactly why the DPDP bug
happened — a wrong heading poisons the vector. Keep the header, validate it.

**Why a distance floor, and why isn't it enough?** Vector search always returns
k results. But the gap is tiny: hardest real answer 0.600, Basel III
(banking, not in the corpus) 0.654. A threshold can't tell "covered" from
"looks covered", so the prompt rule is a second gate.

**Why is the floor before the reranker, not after?** The floor is the refusal
gate, and a cross-encoder's relevance score isn't calibrated to say "nothing
here answers this".

**Why max 2 per section?** One long section split into six parts could fill
every slot. Two, not one, because an obligation and its proviso often sit in
consecutive parts.

**How does framework detection work?** Tokens from framework ids; a naming word
(`dpdp`) scores 2, an instrument word (`act`) scores 1, so "the DPDP Act" picks
the Act over the Rules. Short tokens must be capitalised ("it" vs IT Act). Two
frameworks named = comparison, no scoping. A miss inside a guessed scope falls
back to the whole corpus — a guess never causes a refusal.

**Why condense only after a miss?** Out-of-corpus questions then cost zero LLM
calls. A follow-up like "what about GDPR?" only needs rewriting when it fails
on its own.

### Evaluation

**How do you measure retrieval?** 30 hand-written cases, each naming a phrase
the corpus really contains (checked against the JSON, not taken from what
retrieval returned). Pass = a retrieved passage contains it. No LLM calls.

**Why MRR over mean rank?** Mean rank averages only the cases that passed, so it
*improves* when a hard case fails. MRR over all cases scores a miss as 0. The
harness prints both, and you should lead with MRR.

**What did the reranker buy?** Same index, same cases, only rerank toggled:
Hit@1 77% → 88%, MRR 0.854 → 0.924. The BGE → Cohere gain *can't* be cleanly
attributed, because the chunking changed at the same time — say that.

**Is 30 cases enough?** No. One case ≈ 4 points of Hit@1, so treat single-case
differences as noise.

**What don't you measure?** Faithfulness — whether passage *n* supports the
sentence citing it. Next steps: regex the expected figure out of the answer
(`₹250 crore`, `72 hours`), check framework attribution, then a judge model
(e.g. RAGAS faithfulness) for citation support.

### Engineering

**Hardest extraction traps?** Footnotes numbered 1, 2, 3 on every page looked
like the section sequence restarting; a Companies Act contents page ending in
"Schedules" flipped the parser into schedule mode; DPDP margin-note titles read
as a 2-column table; `[270.` brackets on amended sections; PyMuPDF 2.0
`find_tables()` segfaulting on PCI-DSS (needs `use_layout=False`); small
capitals dropped by the footnote font filter.

**Why Cohere?** Budget was zero. OpenAI is paid. The Gemini free tier counts
every text in a batch toward 1,000/day, so one build would take ~5 days. The
Cohere trial counts per *call* (96 texts = 1), so a build is ~45 calls.
Bonus: no torch, so the deployed app is ~1 GB lighter.

**What stops a model/index mismatch?** The collection stores the embedding model
and dimension it was built with; the app refuses to start against a mismatch
rather than returning unrelated passages.

**Prompt injection?** Honest answer: stopped by the floor, not by detection —
"Ignore all previous instructions…" scores 0.80, far from the corpus. An
injection written in dense compliance language would clear retrieval and meet
only the prompt rule.

**How do you keep citations honest across turns?** Every turn numbers its
sources from `[1]`, but only this turn's sources are in the prompt. A previous
answer's `[2]` pointed at a passage from *its* turn, so replayed as-is it reads
as this turn's `[2]`. `build_prompt` strips markers from earlier answers
(`rag.strip_citations`), so every number the model sees refers to a passage it
can actually read. Then `validate_citations` checks every `[n]` in the reply is
in range.

**Do you send the whole chat history?** Yes, deliberately, so the model keeps
the conversation's context. Two guards make that safe: old citation markers
are stripped, and retrieval never depends on history — `condense` rewrites a
follow-up into a standalone question first. The trade-off is prompt size in a
very long chat; the next step there would be summarising older turns.

**Why not LangGraph?** The flow is linear with a couple of early exits; a
state graph adds a dependency without removing a branch.

### Weaknesses (say them before you're asked)

- Grounding unmeasured (above).
- Chat history is global — not scoped per user; fine locally, not for a shared
  deployment. Also not durable on Streamlit Cloud (ephemeral disk).
- Full chat history goes to the model every turn — a deliberate choice for
  coherence, with prompt size growing in long chats as the cost. (`condense`
  only needs the last 4 messages, because it only feeds retrieval.)
- `uncited` from `validate_citations` is computed but unused.
- OCR'd documents lose font sizes, so footnote/margin handling doesn't apply.
- Free-tier budget caps a public deployment at ~450 questions/month.

**If you had two more weeks?** A faithfulness eval;
hybrid BM25 + vector for exact-figure questions; per-user history on Postgres;
CI running tests + eval.
