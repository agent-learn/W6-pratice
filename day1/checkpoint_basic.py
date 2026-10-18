"""
Checkpointer 最小示例
"""
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class State(TypedDict):
    value:int

def add_one(state:State) -> dict:
    return {"value":state["value"]+1}

builder=StateGraph(State)
builder.add_node("add_one",add_one)
builder.add_edge(START,"add_one")
builder.add_edge("add_one",END)

with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    graph=builder.compile(checkpointer=checkpointer)

    config={"configurable":{"thread_id":"1"}}

    result=graph.invoke({"value":3},config)
    print(f"结果：{result}")

    #state=graph.get_state(config)
    #print(f"状态快照：{state}")
