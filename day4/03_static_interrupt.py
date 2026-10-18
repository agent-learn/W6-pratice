"""
静态中断：interrupt_before / interrupt_after
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

class State(TypedDict):
    messages:list
    result:str

def agent_node(state:State) -> dict:
    return {"messages":state["messages"]+["Agent 思考中..."]}

def tools_node(state:State) -> dict:
    return {"messages":state["messages"]+["工具执行完成"]}

def final_node(state:State) -> dict:
    return {"result":"处理完成"}

builder=StateGraph(State)
builder.add_node("agent",agent_node)
builder.add_node("tools",tools_node)
builder.add_node("final",final_node)

builder.add_edge(START,"agent")
builder.add_edge("agent","tools")
builder.add_edge("tools","final")
builder.add_edge("final",END)

memory=MemorySaver()

graph1=builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

graph2=builder.compile(
    checkpointer=memory,
    interrupt_after=["agent"]
)

config={"configurable":{"thread_id":"static-1"}}

print("=" * 50)
print("静态中断：interrupt_before=['tools']")
print("=" * 50)

result=graph1.invoke({"messages":[],"result":""},config)
state=graph1.get_state(config)
print(f"中断在：{state.next}")

result=graph1.invoke(Command(resume=True),config)
print(f"恢复后结果：{result['result']}")