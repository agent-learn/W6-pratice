"""
恢复时编辑参数
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    content:str
    edited_content:str
    status:str

def draft_content(state:State) -> dict:
    return {"content":"这是原始草稿内容"}

def review_content(state:State) -> dict:
    edited=interrupt({
        "message":"请审核并编辑内容",
        "current_content":state["content"]
    })

    return {"edited_content":edited}

def publish(state:State) -> dict:
    return {
        "status":"published",
        "content":state["edited_content"]
    }

builder=StateGraph(State)
builder.add_node("draft",draft_content)
builder.add_node("review",review_content)
builder.add_node("publish",publish)

builder.add_edge(START,"draft")
builder.add_edge("draft","review")
builder.add_edge("review","publish")
builder.add_edge("publish",END)

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"edit-1"}}

print("第1步：生成草稿并中断")
result=graph.invoke(
    {"content":"","edited_content":"","status":""},
    config
)

state=graph.get_state(config)
print(f"中断状态: {state.next}")

print("\n第2步：恢复并传入编辑后的内容")
edited_text="这是经过人工编辑后的最终内容"
result=graph.invoke(
    Command(resume=edited_text),
    config
)

print(f"最终内容: {result['content']}")
print(f"状态: {result['status']}")

