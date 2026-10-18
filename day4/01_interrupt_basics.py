"""
Interrupt 基础用法
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    action:str
    approved:bool
    result:str

def propose_action(state:State) -> dict:
    print(f"🤖 Agent 提议: {state['action']}")
    return {"action":state["action"]}

def human_approval(state:State) -> dict:
    approved=interrupt(f"是否批准执行 '{state['action']}'?(yes/no)")

    print(f"👤 人类回复：{approved}")
    return {"approved":approved=="yes"}

def execute_action(state:State) -> dict:
    if state["approved"]:
        print(f"✅ 执行：{state['action']}")
        return {"result":f"已执行：{state['action']}"}
    else:
        print(f"❌ 拒绝执行：{state['action']}")
        return {"result":f"已拒绝：{state['action']}"}

builder=StateGraph(State)
builder.add_node("propose",propose_action)
builder.add_node("approval",human_approval)
builder.add_node("execute",execute_action)

builder.add_edge(START,"propose")
builder.add_edge("propose","approval")
builder.add_edge("approval","execute")
builder.add_edge("execute",END)

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"test-1"}}

print("=" * 50)
print("第一次运行（会被中断）")
print("=" * 50)

result=graph.invoke(
    {"action":"删除用户数据","approved":False,"result":""},
    config
)

state=graph.get_state(config)
print(f"\n中断节点：{state.next}")
print(f"中断信息：{state.tasks[0].interrupts if state.tasks else '无'}")
