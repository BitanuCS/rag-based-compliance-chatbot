from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]
while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    results = model.invoke(chat_history)
    chat_history.append(AIMessage(results.content))
    print("AI: ", results.content)
    
# result = model.invoke("Who is the PM of india?")

print(chat_history)
