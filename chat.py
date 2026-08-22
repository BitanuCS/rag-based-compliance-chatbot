"""Model construction and LangSmith run configuration.

Deliberately has no chat loop of its own. An ungrounded assistant answering
compliance questions from the model's memory — no corpus, no citations, no way
to refuse — is the one thing this project exists to prevent, so the only path to
the model runs through rag.build_prompt. For a terminal-side look at what the
corpus holds, `python ingest.py query "..."` searches the index directly.
"""

import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

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


def tracing_enabled():
    """True when LangSmith tracing is switched on via the environment."""
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return flag.strip().lower() in {"1", "true", "yes"}


def tracing_project():
    return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"


def run_config(thread_id, **metadata):
    """LangChain run config that files this turn under a LangSmith thread.

    LangSmith groups runs into a conversation thread by the `session_id`
    metadata key; `thread_id` and `conversation_id` are accepted aliases, so all
    three are set to the same value to stay robust across UI versions.
    """
    return {
        "run_name": "compliance-chat-turn",
        "tags": ["compliance-chat"],
        "metadata": {
            "session_id": thread_id,
            "thread_id": thread_id,
            "conversation_id": thread_id,
            **metadata,
        },
    }


