import os
import uuid

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def build_model(repo_id = DEFAULT_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id = repo_id,
        task = "text-generation"
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


def main():
    model = build_model()
    thread_id = uuid.uuid4().hex
    config = run_config(thread_id)
    chat_history = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    ]
    while True:
        user_input = input("User: ")
        chat_history.append(HumanMessage(content=user_input))
        if user_input == 'exit':
            break
        results = model.invoke(chat_history, config=config)
        chat_history.append(AIMessage(results.content))
        print("AI: ", results.content)

    print(chat_history)


if __name__ == "__main__":
    main()
