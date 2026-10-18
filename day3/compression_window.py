"""
滑动窗口压缩示例
只保留最近 N 条消息
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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
    messages=state["messages"]

    MAX_MESSAGES=6
    if len(messages)>MAX_MESSAGES:
        print(f"[压缩] 从 {len(messages)} 条压缩到 {MAX_MESSAGES} 条")
        recent_messages=messages[-MAX_MESSAGES:]
    else:
        recent_messages=messages

    print(f"[节点执行] 使用 {len(recent_messages)} 条消息")

    response=model.invoke(recent_messages)
    return {"messages":[response]}

builder=StateGraph(ChatState)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

graph=builder.compile()

def main():
    print("=" * 60)
    print("🧪 滑动窗口压缩")
    print("=" * 60)    

    messages=[]
    for i in range(1,8):
        print(f"\n--- 第{i}轮 ---")
        messages.append(HumanMessage(content=f"这是第{i}个问题"))

        result=graph.invoke({"messages":messages})
        response=result["messages"][-1]
        messages.append(response)

        print(f"AI: {response.content[:50]}...")

if __name__ == "__main__":
    main()

