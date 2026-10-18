"""
一个节点内有多个 interrupt
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    step1_result:str
    step2_result:str
    final:str

def multi_step_node(state:State) -> dict:
    answer1=interrupt("问题1：请输入你的名字")
    print(f"回答1：{answer1}")

    answer2=interrupt("问题2：请输入你的邮箱")
    print(f"回答2：{answer2}")

    return {
        "step1_result":answer1,
        "step2_result":answer2,
        "final":f"{answer1} - {answer2}"
    }

builder=StateGraph(State)
builder.add_node("multi_step",multi_step_node)
builder.add_edge(START,"multi_step")
builder.add_edge("multi_step",END)

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"multi-1"}}

result=graph.invoke(
    {"step1_result":"","step2_result":"","final":""},
    config
)
print(f"第1次中断，状态：{graph.get_state(config).next}")

result=graph.invoke(Command(resume="张三"),config)
print(f"第1次恢复后，状态：{graph.get_state(config).next}")

result=graph.invoke(Command(resume="zhangsan@example.com"),config)
print(f"最终结果：{result['final']}")