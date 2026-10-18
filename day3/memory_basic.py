"""
短期记忆 + 长期记忆 组合示例
"""
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from long_term_memory import get_memory_summary, update_user_memory
from memory_tools import MEMORY_TOOLS

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ==================================================
# 1. 定义状态
# ==================================================
class ChatState(TypedDict):
    messages:Annotated[list,add_messages]
    user_id:str

# ==================================================
# 2. 定义节点（带长期记忆注入）
# ==================================================
model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
#model=model.bind_tools(MEMORY_TOOLS)

def call_model(state:ChatState) -> dict:
    user_id=state.get("user_id","unknown")

    memory_summary=get_memory_summary(user_id)

    system_prompt=f"""你是一个贴心的助手。

{memory_summary}

工作原则：
1. 如果用户提到关于自己的新信息（名字、偏好、习惯），调用 save_user_memory 工具保存
2. 回答时参考已知的用户信息
3. 回答要自然、友好"""

    messages=[SystemMessage(content=system_prompt)]+state["messages"]

    print(f"\n[节点执行] 消息数: {len(state['messages'])}")
    print(f"[长期记忆] {memory_summary[:100]}...")

    response=model.invoke(messages)
    return {"messages":[response]}

# ==================================================
# 3. 构建图
# ==================================================
builder=StateGraph(ChatState)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

checkpointer=MemorySaver()
graph=builder.compile(checkpointer=checkpointer)

# ==================================================
# 4. 测试
# ==================================================
def main():
    import json
    if os.path.exists("long_term_memory.json"):
        os.remove("long_term_memory.json")

    print("=" * 60)
    print("📅 会话1（thread_id='session_1'）")
    print("=" * 60)    

    config1={
        "configurable":{"thread_id":"session_1"},
    }

    r1=graph.invoke(
        {
            "messages":[HumanMessage(content="我叫张三，我是一名Python开发者，喜欢简洁的代码风格")],
            "user_id":"user_001",
        },
        config1
    )
    print(f"\nAI: {r1['messages'][-1].content}")

    update_user_memory("user_001","name","张三")
    update_user_memory("user_001","language","Python")
    update_user_memory("user_001","style","简短的代码风格")
    print("\n[已保存长期记忆]")

    print("\n" + "=" * 60)
    print("📅 会话2（thread_id='session_2'，模拟下次再来）")
    print("=" * 60)

    config2={
        "configurable":{"thread_id":"session_2"},
    }

    r2=graph.invoke(
        {
            "messages":[HumanMessage(content="你还记得我吗？我叫什么名字？")],
            "user_id":"user_001",
        },
        config2
    )
    print(f"\nAI: {r2['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("📅 会话3（user_id='user_002'，另一个用户）")
    print("=" * 60)

    r3=graph.invoke(
        {
            "messages":[HumanMessage(content="我叫什么名字？")],
            "user_id":"user_002",
        },
        config1
    )
    print(f"\nAI: {r3['messages'][-1].content}")

if __name__ == "__main__":
    main()
