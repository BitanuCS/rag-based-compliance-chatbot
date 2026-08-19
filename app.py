import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import store
from chat import (
    DEFAULT_REPO_ID,
    DEFAULT_SYSTEM_PROMPT,
    build_model,
    run_config,
    tracing_enabled,
    tracing_project,
)

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


def to_langchain(message):
    if message["role"] == "user":
        return HumanMessage(content=message["content"])
    return AIMessage(content=message["content"])


def new_thread():
    """Mint a thread id for a chat that has not been saved yet.

    Minting up front (rather than at save time) means the very first turn is
    already traced under the thread it belongs to, and the id later becomes the
    conversation's primary key so LangSmith and SQLite agree on one identifier.
    """
    st.session_state.active_id = uuid.uuid4().hex


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

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a compliance question..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    conv_id = st.session_state.active_id
    history = [to_langchain(m) for m in messages]
    prompt = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT), *history, HumanMessage(content=user_input)]

    # Every turn of this chat carries the same thread id, so LangSmith stacks
    # them into one conversation rather than a pile of unrelated runs.
    config = run_config(conv_id, model=MODEL_REPO_ID, turn=len(messages) // 2 + 1)

    with st.chat_message("assistant"):
        try:
            model = get_model(MODEL_REPO_ID)
            reply = st.write_stream(
                chunk.content for chunk in model.stream(prompt, config=config)
            )
        except Exception as exc:  # network/auth/model errors all surface the same way
            st.error(f"Request failed: {exc}")
        else:
            # Persist only a completed turn, creating the conversation on first
            # use under the thread id the trace already used.
            if not saved:
                store.create_conversation(title_from(user_input), conv_id=conv_id)
            store.append_message(conv_id, "user", user_input)
            store.append_message(conv_id, "assistant", reply)
            # Refresh so the sidebar picks up the new/reordered conversation.
            st.rerun()
