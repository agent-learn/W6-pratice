"""
MemorySaver 基础用法
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


# ==================================================
# 1. 定义状态
# ==================================================
class ChatState(TypedDict):
    messages:Annotated[list,add_messages]

# ==================================================
# 2. 定义节点
# ==================================================
model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def can_call(state:ChatState) -> dict:
    print(f"\n[节点执行] 当前消息数: {len(state['messages'])}")
    for msg in state["messages"]:
        print(f"    - {type(msg).__name__}: {msg.content[:40]}...")

    response=model.invoke(state["messages"])
    return {"messages":[response]}

# ==================================================
# 3. 构建图
# ==================================================
builder=StateGraph(ChatState)
builder.add_node("model",can_call)
builder.add_edge(START,"model")
builder.add_edge("model",END)

# ==================================================
# 4. 用 MemorySaver 编译
# ==================================================
checkpointer=MemorySaver()
graph=builder.compile(checkpointer=checkpointer)

# ==================================================
# 5. 多轮对话
# ==================================================
def main():
    config={"configurable":{"thread_id":"user_001"}}

    print("=" * 60)
    print("🧪 MemorySaver 多轮对话")
    print("=" * 60)

    print("\n--- 第1轮 ---")
    r1=graph.invoke(
        {"messages":[HumanMessage(content="我叫张三，是一名Python开发者")]},
        config
    )
    print(f"AI: {r1['messages'][-1].content}")

    print("\n--- 第2轮 ---")
    r2=graph.invoke(
        {"messages":[HumanMessage(content="我平时喜欢用什么编程语言？")]},
        config
    )
    print(f"AI: {r2['messages'][-1].content}")

    print("\n--- 第3轮 ---")
    r3=graph.invoke(
        {"messages":[HumanMessage(content="我叫什么名字？")]},
        config
    )
    print(f"AI: {r3['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("📊 最终状态")
    print("=" * 60)
    state=graph.get_state(config)
    print(f"消息总数：{len(state.values['messages'])}")

    checkpoints=list(graph.get_state_history(config))
    print(f"检查点总数：{len(checkpoints)}")
    for i,cp in enumerate(checkpoints[:5]):
        source=cp.metadata.get('source','?')
        msg_count=len(cp.values['messages'])
        print(f"  [{i}] source={source},消息数={msg_count}")

    print("\n" + "=" * 60)
    print("🔄 模拟重启：创建新的 MemorySaver")
    print("=" * 60)   

    new_checkpointer=MemorySaver()
    new_graph=builder.compile(checkpointer=new_checkpointer)

    r_restart=new_graph.invoke(
        {"messages":[HumanMessage(content="我叫什么名字？")]},
        config
    )
    print(f"重启后 AI: {r_restart['messages'][-1].content}")

if __name__ == "__main__":
    main()
