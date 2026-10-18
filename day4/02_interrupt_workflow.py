"""
Interrupt 完整工作流：中断 → 审查 → 恢复
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    task:str
    draft:str
    feedback:str
    final:str

def create_draft(state:State) -> dict:
    draft=f"关于「{state['task']}」的初稿内容..."
    print(f"📝 生成初稿：{draft}")
    return {"draft":draft}

def review_draft(state:State) -> dict:
    feedback=interrupt({
        "message":"请审核以下初稿",
        "draft":state["draft"]
    })

    print(f"👤 审核反馈：{feedback}")
    return {"feedback":feedback}

def finalize(state:State) -> dict:
    final=f"{state['draft']}\n\n[根据反馈修改：{state['feedback']}]"
    print(f"✅ 最终稿：{final}")
    return {"final":final}

builder=StateGraph(State)
builder.add_node("draft",create_draft)
builder.add_node("review",review_draft)
builder.add_node("finalize",finalize)

builder.add_edge(START,"draft")
builder.add_edge("draft","review")
builder.add_edge("review","finalize")
builder.add_edge("finalize",END)

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"review-1"}}

print("=" * 50)
print("第1步：运行到中断点")
print("=" * 50)

result=graph.invoke(
    {"task":"写一份项目计划","draft":"","feedback":"","final":""},
    config
)

state=graph.get_state(config)
print(f"\n图已暂停")
print(f"待执行节点：{state.next}")
if state.tasks and state.tasks[0].interrupts:
    print(f"中断负载：{state.tasks[0].interrupts[0].value}")

print("\n" + "=" * 50)
print("第2步：人类审查后恢复执行")
print("=" * 50)

result=graph.invoke(
    Command(resume="需要增加时间线部分"),
    config
)

print(f"\n最终结果：{result['final']}")
