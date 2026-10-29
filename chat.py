"""Model construction and per-turn run configuration.

Deliberately has no chat loop of its own. An ungrounded assistant answering
compliance questions from the model's memory — no corpus, no citations, no way
to refuse — is the one thing this project exists to prevent, so the only path to
the model runs through rag.build_prompt. For a terminal-side look at what the
corpus holds, `python ingest.py query "..."` searches the index directly.
"""

import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from observability import langchain_callbacks

load_dotenv()

DEFAULT_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"


# Ceiling on a single reply. Bounds the cost of any one turn and stops a runaway
# generation streaming indefinitely into the UI. Cited compliance answers run a few
# hundred tokens, so this is headroom rather than a constraint.
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))


def build_model(repo_id = DEFAULT_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id = repo_id,
        task = "text-generation",
        max_new_tokens = MAX_NEW_TOKENS,
    )
    return ChatHuggingFace(llm = llm)


def run_config(thread_id, **metadata):
    """LangChain run config for one LLM call inside a chat turn.

    The Langfuse callback records the call — prompt, completion, token usage,
    latency — as a generation nested under the turn's trace (see
    observability.trace_turn). LangSmith needs no callback: LangChain traces to
    it from the environment on its own.

    The conversation id goes in under every key either tool groups by.
    `langfuse_session_id` is what the Langfuse callback reads; LangSmith threads
    group on `session_id`, with `thread_id` and `conversation_id` as accepted
    aliases. They repeat what trace_turn already sets, so a call made outside a
    turn still lands in the right session and thread.
    """
    return {
        "run_name": "compliance-chat-turn",
        "tags": ["compliance-chat"],
        "callbacks": langchain_callbacks(),
        "metadata": {
            "langfuse_session_id": thread_id,
            "session_id": thread_id,
            "thread_id": thread_id,
            "conversation_id": thread_id,
            **metadata,
        },
    }
