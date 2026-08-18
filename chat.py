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


def main():
    model = build_model()
    chat_history = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    ]
    while True:
        user_input = input("User: ")
        chat_history.append(HumanMessage(content=user_input))
        if user_input == 'exit':
            break
        results = model.invoke(chat_history)
        chat_history.append(AIMessage(results.content))
        print("AI: ", results.content)

    print(chat_history)


if __name__ == "__main__":
    main()
