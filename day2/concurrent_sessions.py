"""
模拟多用户并发对话
"""
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

class ChatState(TypedDict):
    messages:Annotated[list,add_messages]

model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def call_model(state:ChatState) -> dict:
    response=model.invoke(state["messages"])
    return {"messages":[response]}

builder=StateGraph(ChatState)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

checkpointer=MemorySaver()
graph=builder.compile(checkpointer=checkpointer)

def simulate_multi_user():
    sessions={
        "alice":{"configurable":{"thread_id":"alice"}},
        "bob":{"configurable":{"thread_id":"bob"}},
        "charlie":{"configurable":{"thread_id":"charlie"}},
    }

    interactions=[
        ("alice", "我叫Alice，我喜欢猫"),
        ("bob", "我叫Bob，我喜欢狗"),
        ("charlie", "我叫Charlie，我喜欢鸟"),
        ("alice", "我喜欢什么动物？"),
        ("bob", "我喜欢什么动物？"),  
        ("charlie", "我喜欢什么动物？"),  
    ]

    print("=" * 60)
    print("🧪 多用户交替对话")
    print("=" * 60)

    for user,message in interactions:
        config=sessions[user]
        result=graph.invoke(
            {"messages":[HumanMessage(content=message)]},
            config
        )
        reply=result["messages"][-1].content
        print(f"\n[{user}] {message}")
        print(f"  AI: {reply}")

if __name__ == "__main__":
    simulate_multi_user()

