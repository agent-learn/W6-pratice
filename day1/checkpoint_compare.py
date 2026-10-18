import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",".env"))

class ChatState(TypedDict):
    messages:Annotated[list,add_messages]

def create_chatbot(checkpointer=None):
    model=ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )

    def call_model(state:ChatState) -> dict:
        print(f"\n[节点执行] 当前消息数: {len(state['messages'])}")
        for msg in state["messages"]:
            print(f"   - {type(msg).__name__}: {msg.content[:30]}...")

        response=model.invoke(state["messages"])
        return {"messages":[response]}

    builder=StateGraph(ChatState)
    builder.add_node("model",call_model)
    builder.add_edge(START,"model")
    builder.add_edge("model",END)

    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()

print("=" * 60)
print("实验1：没有 Checkpointer（每次失忆）")
print("=" * 60)

bot_no_memory=create_chatbot()

r1=bot_no_memory.invoke({"messages":[HumanMessage(content="我叫张三")]})
print(f"用户: 我叫张三")
print(f"AI: {r1['messages'][-1].content}")

r2=bot_no_memory.invoke({"messages":[HumanMessage(content="我叫什么名字？")]})
print(f"\n用户: 我叫什么名字？")
print(f"AI: {r2['messages'][-1].content}")

print("\n" + "=" * 60)
print("实验2：有 Checkpointer（记住历史）")
print("=" * 60)

with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    bot_with_memory=create_chatbot(checkpointer=checkpointer)
    config={"configurable":{"thread_id":"user_001"}}

    r1=bot_with_memory.invoke(
        {"messages":[HumanMessage(content="我叫张三")]},
        config
    )
    print(f"用户: 我叫张三")
    print(f"AI: {r1['messages'][-1].content}")

    r2=bot_with_memory.invoke(
        {"messages":[HumanMessage(content="我叫什么名字？")]},
        config
    )
    print(f"\n用户: 我叫什么名字？")
    print(f"AI: {r2['messages'][-1].content}")

