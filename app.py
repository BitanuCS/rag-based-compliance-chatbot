import os
import uuid

import streamlit as st
from dotenv import load_dotenv

import rag
import store
from chat import DEFAULT_REPO_ID, build_model, run_config, tracing_enabled, tracing_project
from vectorstore import indexed_frameworks

load_dotenv()

MODEL_REPO_ID = os.getenv("HF_MODEL", DEFAULT_REPO_ID)
TITLE_MAX_CHARS = 34

st.set_page_config(page_title="Compliance Chat", page_icon="⚖️")

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
    backed the answer, so the label follows what the answer actually used and
    passages it never referenced are marked as such.

    A refusal is treated as using nothing, even when it contains [n] markers — a
    refusal routinely cites while *describing* what the sources cover ("the sources
    cover digital lending [1]"), which is the opposite of drawing on them. Counting
    those would label a refusal "2 sources" and invite the reader to conclude from
    them exactly what the refusal said could not be concluded.

    Collapsed by default: the point is that every claim *can* be checked, not
    that the reader must wade through 6 clauses to read one answer.
    """
    if not sources:
        return

    cited, _, _ = rag.validate_citations(answer, len(sources))
    # Only markers pointing at a source we actually supplied may be counted, or a
    # hallucinated [9] inflates the header above the number of passages listed.
    cited = [n for n in cited if 1 <= n <= len(sources)]
    if rag.is_refusal(answer):
        cited = []

    if not cited:
        label = f"🔍 {len(sources)} passages searched — none used in this answer"
    elif len(cited) == len(sources):
        label = f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''}"
    else:
        # Always "N of M", never a bare count: the panel lists all M passages, so a
        # header saying "2 sources" above six entries reads as a bug.
        label = f"📎 {len(cited)} of {len(sources)} sources cited"

    with st.expander(label):
        for source in sources:
            used = source["n"] in cited
            # Explains why the header count can be lower than the list length.
            tag = "" if used else " · _not cited_"
            st.markdown(
                f"**[{source['n']}] {source['framework_id']}** · {source['pages']} "
                f"· distance `{source['distance']}`{tag}"
            )
            st.caption(source["breadcrumb"])
            st.text(source["text"][:1500])
            if source.get("source_url"):
                st.caption(f"[{source['framework_name']}]({source['source_url']})")
            st.divider()


if "active_id" not in st.session_state:
    new_thread()

# The DB is shared across browser sessions, so a saved chat may have been
# deleted elsewhere since this session last rendered.
saved = store.conversation_exists(st.session_state.active_id)
conversations = store.list_conversations()
messages = store.load_messages(st.session_state.active_id) if saved else []

with st.sidebar:
    st.title("⚖️ Compliance Chat")

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

    st.divider()
    if tracing_enabled():
        st.caption(f"🔎 LangSmith: `{tracing_project()}`")
        st.caption(f"thread `{st.session_state.active_id[:12]}`")
    else:
        st.caption("🔎 LangSmith: off")

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
            model = get_model(MODEL_REPO_ID)

            with st.spinner("Searching the corpus..."):
                # A follow-up like "what about GDPR?" is meaningless to a vector
                # search on its own, so history is folded in before retrieving.
                search_query = rag.condense(model, user_input, history, config)
                hits, best = rag.retrieve(search_query, framework=framework)

            if not hits:
                # Nothing cleared the distance floor. Answering anyway would mean
                # writing around irrelevant regulation, so refuse and show how
                # close the nearest match came.
                reply = rag.NO_CONTEXT
                if best is not None:
                    reply += f"\n\n*(nearest match: distance {best:.3f}, floor {rag.MAX_DISTANCE})*"
                st.markdown(reply)
                sources = []
            else:
                sources = rag.citations(hits)
                prompt = rag.build_prompt(user_input, hits, history)
                reply = st.write_stream(
                    chunk.content for chunk in model.stream(prompt, config=config)
                )
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
            st.error(f"Request failed: {exc}")
        else:
            # Persist only a completed turn, creating the conversation on first
            # use under the thread id the trace already used.
            if not saved:
                store.create_conversation(title_from(user_input), conv_id=conv_id)
            store.append_message(conv_id, "user", user_input)
            store.append_message(conv_id, "assistant", reply, sources=sources)
            # Refresh so the sidebar picks up the new/reordered conversation.
            st.rerun()
