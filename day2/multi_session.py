"""
多会话隔离示例
用 thread_id 实现不同用户的对话隔离
"""
import os
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ==================================================
# 1. 定义状态和节点
# ==================================================
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
    return {'messages':[response]}

# ==================================================
# 2. 构建图
# ==================================================
builder=StateGraph(ChatState)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

checkpointer=MemorySaver()
graph=builder.compile(checkpointer=checkpointer)

# ==================================================
# 3. 模拟多个用户
# ==================================================
def chat(thread_id:str,message:str) -> str:
    config={"configurable":{"thread_id":thread_id}}
    result=graph.invoke(
        {"messages":[HumanMessage(content=message)]},
        config
    )
    return result["messages"][-1].content

def main():
    print("=" * 60)
    print("🧪 多会话隔离演示")
    print("=" * 60)    

    print("\n--- 用户A (thread_id='user_A') ---")
    print(f"A: 我叫张三")
    print(f"AI: {chat('user_A','我叫张三')}")

    print(f"\nA: 我是一名Python开发者")
    print(f"AI: {chat('user_A','我是一名Python开发者')}")

    print("\n--- 用户B (thread_id='user_B') ---")
    print(f"B: 我叫李四")
    print(f"AI: {chat('user_B','我叫李四')}")

    print(f"\nB: 我是一名Java开发者")
    print(f"AI: {chat('user_B','我是一名Java开发者')}")

    print("\n--- 用户A 回到自己的对话 ---")
    print(f"A: 我叫什么名字？")
    print(f"AI: {chat('user_A','我叫什么名字？')}")

    print(f"\nA: 我用什么编程语言？")
    print(f"AI: {chat('user_A','我用什么编程语言？')}")

    print("\n--- 用户B 回到自己的对话 ---")
    print(f"B: 我叫什么名字？")
    print(f"AI: {chat('user_B','我叫什么名字？')}")

    print("\n" + "=" * 60)
    print("📊 各会话的检查点")
    print("=" * 60)

    for tid in ["user_A","user_B"]:
        config={"configurable":{"thread_id":tid}}
        checkpoints=list(graph.get_state_history(config))
        state=graph.get_state(config)
        print(f"\n  thread_id='{tid}':")
        print(f"    消息数: {len(state.values['messages'])}")
        print(f"    检查点数: {len(checkpoints)}")

if __name__ == "__main__":
    main()
