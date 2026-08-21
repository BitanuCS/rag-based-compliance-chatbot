"""Retrieval and grounding: turn a question into a cited, corpus-backed prompt.

Kept out of app.py so the retrieval policy is testable from the CLI and reusable
from a future API layer, matching how store.py and vectorstore.py are split.

The policy has three parts, and all three matter:

1. **A distance floor.** Vector search always returns k results, however bad the
   match — a chocolate-cake question returns Labour Codes at distance 0.59, while
   a real KYC question hits 0.23. Without a floor the model gets handed irrelevant
   regulation and writes a confident answer around it, which is worse than no
   answer at all.
2. **A grounded prompt.** The model answers from the retrieved context or says it
   cannot, and is told never to fall back on its own knowledge of the law.
3. **Numbered sources.** Every context block is labelled [1], [2]… and the model
   cites those markers inline, so each claim traces to a framework, section and
   page the reader can check against the source PDF.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from vectorstore import display_text, get_store

# Cosine distance, so lower is closer. Measured on this corpus: genuine hits land
# around 0.23-0.30 and off-topic questions bottom out near 0.59. 0.45 sits in the
# gap. Raise it to be more permissive, lower it to demand tighter matches.
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.45"))
RETRIEVAL_K = int(os.getenv("RAG_K", "6"))

SYSTEM_PROMPT = """You are an experienced and professional Compliance Engineer \
advising on Indian and international regulatory frameworks. You are precise, \
measured, and careful about the difference between what a regulation requires and \
what is merely good practice. You give professional judgements, and you are explicit \
about their limits.

Answer using ONLY the numbered sources provided below. These sources are the only \
evidence you have. Your professional standing rests on never stating a regulatory \
obligation you cannot point to in them.

Rules:
- Cite the source number inline, like [1] or [2][3], immediately after each claim \
it supports. Every factual statement needs a citation.
- If the sources do not answer the question, reply exactly: "This is out of my \
knowledge." Then state briefly what the sources you were given do cover. Do not \
fill the gap from memory, and do not guess from professional experience.
- Never cite a source number that was not provided to you.
- Quote the regulation's own wording for specific obligations, thresholds and \
deadlines rather than paraphrasing them.
- If sources disagree or come from different frameworks, say which framework each \
position belongs to.
- Be concise and precise. This is regulatory text: exact wording matters more than \
readable prose."""

# Rewrites a follow-up into a standalone question. Without this, "what about GDPR?"
# retrieves on those four words alone and finds nothing useful, because the subject
# of the question lives in the previous turn.
CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone question that \
makes sense without the conversation history. Keep the user's original wording and \
intent wherever possible; only add the context needed to make it self-contained. \
Reply with the rewritten question and nothing else."""

# Worded to match the refusal the system prompt asks for, so the two ways of
# reaching "I don't know" — nothing cleared the distance floor, and the model
# judging the retrieved sources insufficient — read identically to the user.
NO_CONTEXT = (
    "**This is out of my knowledge.**\n\n"
    "Nothing in the compliance corpus I have indexed answers this question.\n\n"
    "The corpus covers 25 frameworks — RBI, SEBI and IRDAI directions, the DPDP Act "
    "and Rules, the IT Act, PCI-DSS, GDPR, ISO 27001, the Companies Act and the "
    "Labour Codes. Try naming the framework or the obligation you have in mind."
)


def condense(model, question, history, config=None):
    """Fold conversation history into a standalone retrieval query.

    Only used when there is history to fold in — the first question of a chat is
    already standalone. Falls back to the raw question if the rewrite fails, since
    a degraded query still retrieves better than a crashed one.
    """
    if not history:
        return question

    # Recent turns only: older ones rarely change what the follow-up refers to,
    # and they push up latency and cost on every single turn.
    recent = history[-4:]
    prompt = [
        SystemMessage(content=CONDENSE_PROMPT),
        *recent,
        HumanMessage(content=f"Follow-up question: {question}"),
    ]
    try:
        rewritten = model.invoke(prompt, config=config).content.strip()
    except Exception:
        return question
    return rewritten or question


def retrieve(question, k=RETRIEVAL_K, framework=None, max_distance=MAX_DISTANCE):
    """Nearest chunks that clear the distance floor.

    Returns (hits, best_distance). best_distance is reported even when everything
    is filtered out, so a near-miss is debuggable rather than an unexplained
    refusal.
    """
    from vectorstore import build_filter

    scored = get_store().similarity_search_with_score(
        question, k=k, filter=build_filter(framework)
    )
    if not scored:
        return [], None

    best = scored[0][1]
    return [(doc, dist) for doc, dist in scored if dist <= max_distance], best


def citations(hits):
    """Source list for display and for persistence, numbered as the model sees it."""
    out = []
    for number, (doc, distance) in enumerate(hits, start=1):
        meta = doc.metadata
        pages = f"p{meta['page_start']}"
        if meta["page_end"] != meta["page_start"]:
            pages += f"–{meta['page_end']}"
        out.append(
            {
                "n": number,
                "framework_id": meta["framework_id"],
                "framework_name": meta["framework_name"],
                "breadcrumb": meta["breadcrumb"],
                "pages": pages,
                "source_url": meta.get("source_url", ""),
                "distance": round(distance, 4),
                "text": display_text(doc),
            }
        )
    return out


def format_context(hits):
    """The numbered evidence block the model reads.

    Each block leads with its breadcrumb so the model can attribute a claim to a
    named framework and section rather than to an anonymous wall of text.
    """
    blocks = []
    for number, (doc, _) in enumerate(hits, start=1):
        meta = doc.metadata
        blocks.append(
            f"[{number}] {meta['framework_name']}\n"
            f"Location: {meta['breadcrumb']} (page {meta['page_start']})\n\n"
            f"{display_text(doc)}"
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(question, hits, history=None):
    """Grounded prompt: system rules, prior turns, then evidence plus the question.

    The evidence rides with the current question rather than as a separate system
    message so that on a later turn it is clear which question each context block
    was retrieved for.
    """
    grounded = (
        f"Sources:\n\n{format_context(hits)}\n\n"
        f"---\n\nQuestion: {question}"
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *(history or []),
        HumanMessage(content=grounded),
    ]


def to_langchain(message):
    """Stored message dict -> LangChain message."""
    if message["role"] == "user":
        return HumanMessage(content=message["content"])
    return AIMessage(content=message["content"])
