import logging
import os
import uuid

# First, and before the project imports below: _bootstrap swaps in a newer
# sqlite for chromadb where the host needs it, and copies Streamlit secrets into
# os.environ. Both have to happen before `import rag` pulls in vectorstore.py,
# which reads its configuration and opens Chroma at module scope.
import _bootstrap

_bootstrap.load_secrets_into_env()

import streamlit as st
from dotenv import load_dotenv

import rag
import store
from chat import DEFAULT_REPO_ID, build_model, run_config, tracing_enabled
from vectorstore import indexed_frameworks

# Fills in anything secrets did not provide; does not override real env vars.
load_dotenv()

MODEL_REPO_ID = os.getenv("HF_MODEL", DEFAULT_REPO_ID)
TITLE_MAX_CHARS = 34

st.set_page_config(page_title="Compliance Chat", page_icon="⚖️")

# History rows are one line each. Streamlit centres and wraps button labels, which
# turns a list of chats into a stack of two-line blocks you have to read rather
# than scan. Scoped by the `open-<id>` key so it touches only the history buttons.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [class*="st-key-open-"] button {
        justify-content: flex-start;
        text-align: left;
    }
    section[data-testid="stSidebar"] [class*="st-key-open-"] button div,
    section[data-testid="stSidebar"] [class*="st-key-open-"] button p {
        min-width: 0;
        max-width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cached so the endpoint isn't rebuilt on every Streamlit rerun.
get_model = st.cache_resource(show_spinner=False)(build_model)

store.init_db()


def title_from(text):
    text = " ".join(text.split())
    if len(text) <= TITLE_MAX_CHARS:
        return text
    return text[:TITLE_MAX_CHARS].rstrip() + "…"


def new_thread():
    """Mint a thread id for a chat that has not been saved yet.

    Minting up front (rather than at save time) means the very first turn is
    already traced under the thread it belongs to, and the id later becomes the
    conversation's primary key so LangSmith and SQLite agree on one identifier.
    """
    st.session_state.active_id = uuid.uuid4().hex


def render_citations(sources, answer):
    """Show the evidence behind an answer, labelled by what the answer actually used.

    Retrieved is not the same as cited. Calling a passage a "source" claims it
    backed the answer, so only the passages the answer actually referenced are
    listed; the rest were search results, not evidence, and showing them only
    invites the reader to treat unused text as support.

    A refusal is treated as using nothing, even when it contains [n] markers — a
    refusal routinely cites while *describing* what the sources cover ("the sources
    cover digital lending [1]"), which is the opposite of drawing on them. Counting
    those would label a refusal "2 sources" and invite the reader to conclude from
    them exactly what the refusal said could not be concluded.

    The references themselves sit directly under the answer, always visible: a
    citation the reader has to go looking for is doing half its job. The full
    passage text stays collapsed behind them — the point is that every claim *can*
    be checked, not that the reader must wade through six clauses to read one
    answer.
    """
    if not sources:
        return

    cited, _, _ = rag.validate_citations(answer, len(sources))
    # Only markers pointing at a source we actually supplied may be counted, or a
    # hallucinated [9] inflates the header above the number of passages listed.
    cited = [n for n in cited if 1 <= n <= len(sources)]
    if rag.is_refusal(answer):
        cited = []

    # Numbers stay as the model saw them, so [3] in the answer still finds [3] here
    # even when the gaps are gone.
    shown = [source for source in sources if source["n"] in cited]
    if not shown:
        return

    groups = rag.group_references(shown)
    st.markdown(f"**References** · {len(groups)} source{'s' if len(groups) != 1 else ''}")
    for place, numbers in groups:
        markers = "".join(f"[{n}]" for n in numbers)
        # Framework name over id: the reference line is for the reader, and
        # "Digital Personal Data Protection Act, 2023" identifies a document in a
        # way "DPDP_ACT_2023" does not.
        line = f"**{markers}** {place['framework_name']}"
        if place["location"]:
            line += f" · {place['location']}"
        line += f" · {place['pages']}"
        if place["source_url"]:
            line += f" · [source ↗]({place['source_url']})"
        st.markdown(line)

    with st.expander("📎 Show cited passages"):
        for source in shown:
            st.markdown(
                f"**[{source['n']}] {source['framework_id']}** · {source['pages']} "
                f"· distance `{source['distance']}`"
            )
            st.caption(source["breadcrumb"])
            st.text(source["text"][:1500])
            st.divider()


if "active_id" not in st.session_state:
    new_thread()

# The DB is shared across browser sessions, so a saved chat may have been
# deleted elsewhere since this session last rendered.
saved = store.conversation_exists(st.session_state.active_id)
conversations = store.list_conversations()
messages = store.load_messages(st.session_state.active_id) if saved else []

with st.sidebar:
    st.title("⚖️ Compliance Assistant")

    if st.button("➕  New chat", use_container_width=True):
        # Already on an unsaved blank chat? Its thread id is still unused.
        if saved:
            new_thread()
            st.rerun()

    st.divider()

    # Narrowing retrieval to one framework is the difference between "what does
    # the law say" and "what does DPDP say" on questions several frameworks touch.
    scope = st.selectbox(
        "Search scope",
        ["All frameworks", *indexed_frameworks()],
        help="Restrict retrieval to a single framework.",
    )
    framework = None if scope == "All frameworks" else scope

    st.divider()
    st.caption("History")

    if not conversations:
        st.caption("No saved chats yet.")

    for conv in conversations:
        row, delete_col = st.columns([5, 1], gap="small")
        is_active = conv["id"] == st.session_state.active_id
        if row.button(
            conv["title"],
            key=f"open-{conv['id']}",
            # The row clips to one line, so the full title lives in the tooltip.
            help=conv["title"],
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_id = conv["id"]
            st.rerun()
        if delete_col.button("🗑", key=f"del-{conv['id']}", help="Delete chat"):
            store.delete_conversation(conv["id"])
            if is_active:
                new_thread()
            st.rerun()

if not messages:
    st.markdown("### How can I help with compliance today?")
    st.caption(
        "Answers come only from the indexed corpus — 25 frameworks, 2,602 sections — "
        "and every claim is cited back to a section and page."
    )

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_citations(message.get("sources"), message["content"])

if user_input := st.chat_input(
    "Ask a compliance question...",
    # First line of defence, enforced by the browser. check_input() repeats it
    # server-side because this widget is not the only caller of the RAG path.
    max_chars=rag.MAX_INPUT_CHARS,
):
    with st.chat_message("user"):
        st.markdown(user_input)

    rejection = rag.check_input(user_input)
    if rejection:
        # Nothing is persisted: a rejected input never became a turn.
        with st.chat_message("assistant"):
            st.warning(rejection)
        st.stop()

    conv_id = st.session_state.active_id
    history = [rag.to_langchain(m) for m in messages]

    # Every turn of this chat carries the same thread id, so LangSmith stacks
    # them into one conversation rather than a pile of unrelated runs.
    config = run_config(conv_id, model=MODEL_REPO_ID, turn=len(messages) // 2 + 1)

    with st.chat_message("assistant"):
        try:
            model = None

            with st.spinner("Searching the corpus..."):
                # Retrieval runs on the raw question first, before any model is
                # touched. A question the corpus cannot answer is answered by a
                # constant, and reaching a constant through a model call means
                # paying a model to say nothing.
                # An explicit sidebar scope wins; otherwise a question that names
                # a framework is treated as asking about that framework.
                scope_id = framework or rag.detect_framework(user_input)
                hits, best = rag.retrieve(user_input, framework=scope_id)

                if not hits and scope_id and not framework:
                    # Detection is a guess, so it is never allowed to cause a
                    # refusal: a miss inside the guessed framework falls back to
                    # the whole corpus rather than reporting nothing found.
                    hits, best = rag.retrieve(user_input)

                if not hits and history:
                    # The one case where a miss is not yet a refusal: a follow-up
                    # like "what about GDPR?" is meaningless to a vector search on
                    # its own because its subject lives in the previous turn. The
                    # condense call is spent only here, where it can still turn a
                    # refusal into an answer — never to confirm one.
                    model = get_model(MODEL_REPO_ID)
                    search_query = rag.condense(model, user_input, history, config)
                    if search_query != user_input:
                        hits, best = rag.retrieve(
                            search_query,
                            framework=framework or rag.detect_framework(search_query),
                        )

            if not hits:
                # Nothing cleared the distance floor. Answering anyway would mean
                # writing around irrelevant regulation.
                reply = rag.NO_CONTEXT
                if best is not None and tracing_enabled():
                    # Debug detail, and only in dev: how close the nearest match
                    # came is what tells you whether the floor is set too tight.
                    reply += f"\n\n*(nearest match: distance {best:.3f}, floor {rag.MAX_DISTANCE})*"
                st.markdown(reply)
                sources = []
            else:
                model = model or get_model(MODEL_REPO_ID)
                sources = rag.citations(hits)
                prompt = rag.build_prompt(user_input, hits, history)
                # Streamed into a placeholder so the finished text can replace the
                # streamed one: the reserved refusal sentence can only be spotted
                # once the answer it contradicts has been written.
                holder = st.empty()
                with holder:
                    reply = st.write_stream(
                        chunk.content for chunk in model.stream(prompt, config=config)
                    )
                reply = rag.strip_stray_refusal(reply)

                if rag.is_refusal(reply):
                    # The model read the passages and judged them insufficient.
                    # That is the same outcome as retrieving nothing, so it reads
                    # identically: one wording for "the corpus does not answer
                    # this", never a second one the reader has to interpret.
                    reply = rag.NO_CONTEXT
                    sources = []
                holder.markdown(reply)

                # Verify after the fact rather than trusting the prompt: a marker
                # pointing at a source that was never supplied is unverifiable by
                # the reader, which defeats the point of citing at all.
                _, invalid, _ = rag.validate_citations(reply, len(sources))
                if invalid:
                    st.warning(
                        "⚠️ This answer references "
                        + ", ".join(f"[{n}]" for n in invalid)
                        + f", which {'was' if len(invalid) == 1 else 'were'} not among "
                        f"the {len(sources)} sources retrieved. Treat "
                        f"{'that claim' if len(invalid) == 1 else 'those claims'} as unverified."
                    )
                render_citations(sources, reply)
        except Exception as exc:  # network/auth/model errors all surface the same way
            # The detail goes to the server log, not the page: an endpoint error
            # can carry the request URL and headers, and a compliance officer can
            # do nothing with a stack trace anyway.
            logging.exception("answer failed for conversation %s", conv_id)
            st.error(
                "Something went wrong reaching the model. Try again — if it keeps "
                "failing, check the server log."
            )
            st.caption(f"`{type(exc).__name__}`")
        else:
            # Persist only a completed turn, creating the conversation on first
            # use under the thread id the trace already used.
            if not saved:
                store.create_conversation(title_from(user_input), conv_id=conv_id)
            store.append_message(conv_id, "user", user_input)
            store.append_message(conv_id, "assistant", reply, sources=sources)
            # Refresh so the sidebar picks up the new/reordered conversation.
            st.rerun()
