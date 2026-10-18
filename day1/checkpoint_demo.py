"""
Checkpointer 完整 Demo
多轮对话 Agent + 状态查看 + 时间旅行
"""
import os
import json
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",".env"))

# ==================================================
# 1. 定义状态
# ==================================================
class ChatState(TypedDict):
    messages:Annotated[list,add_messages]
    turn_count:int

# ==================================================
# 2. 定义节点
# ==================================================
def call_model(state:ChatState) -> dict:
    model=ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )

    response=model.invoke(state["messages"])
    turn_count=state.get("turn_count",0)+1

    print(f"\n  [执行] 第 {turn_count} 轮 | 消息数: {len(state['messages'])}")

    return {
        "messages":[response],
        "turn_count":turn_count,
    }

# ==================================================
# 3. 构建图
# ==================================================
def build_graph(checkpointer):
    builder=StateGraph(ChatState)
    builder.add_node("model",call_model)
    builder.add_edge(START,"model")
    builder.add_edge("model",END)
    return builder.compile(checkpointer=checkpointer)

# ==================================================
# 4. 辅助函数
# ==================================================
def print_state(state):
    """打印状态快照"""
    print(f"\n  状态快照:")
    print(f"    消息数: {len(state.values['messages'])}")
    print(f"    轮次: {state.values.get('turn_count', 0)}")
    print(f"    下一个节点: {state.next}")

def print_checkpoints(graph,config,limit=5):
    print(f"\n  📜 历史检查点:")
    checkpoints=list(graph.get_state_history(config))
    print(f"\n实际检查点总数: {len(checkpoints)}")
    for i,cp in enumerate(checkpoints[:limit]):
        msg_count=len(cp.values['messages'])
        print(f"    [{i}] checkpoint_id={cp.config['configurable'].get('checkpoint_id','N/A')[:8]}... | 消息数: {msg_count}")

    for cp in checkpoints:
        source = cp.metadata.get('source')
        step = cp.metadata.get('step')
        msg_count = len(cp.values.get('messages', []))
        print(f"source={source}, step={step}, 消息数={msg_count}")

# ==================================================
# 5. 主流程
# ==================================================
def main():
    with SqliteSaver.from_conn_string("chat_history.sqlite") as checkpointer:
        graph=build_graph(checkpointer)
        config={"configurable":{"thread_id":"demo_001"}}

        print("=" * 60)
        print("🧪 Checkpointer Demo：多轮对话")
        print("=" * 60)

        print("\n--- 第1轮 ---")
        r1=graph.invoke(
            {"messages":[HumanMessage(content="我叫张三，是一名Python开发者")]},
            config
        )
        print(f"  AI: {r1['messages'][-1].content}")
        print_state(graph.get_state(config))

        print("\n--- 第2轮 ---")
        r2=graph.invoke(
            {"messages":[HumanMessage(content="我平时喜欢用什么编程语言？")]},
            config
        )
        print(f"  AI: {r2['messages'][-1].content}")
        print_state(graph.get_state(config))

        print("\n--- 第3轮 ---")
        r3 = graph.invoke(
            {"messages": [HumanMessage(content="我叫什么名字？")]},
            config
        )
        print(f"  AI: {r3['messages'][-1].content}")

        print("\n" + "=" * 60)
        print("📜 历史检查点")
        print("=" * 60)
        print_checkpoints(graph,config)

        print("\n" + "=" * 60)
        print("⏰ 时间旅行：回放第1轮之后的检查点")
        print("=" * 60)

        checkpoints=list(graph.get_state_history(config))
        if len(checkpoints)>1:
            old_cp=checkpoints[1]
            print(f"  回放到: {old_cp.config['configurable'].get('checkpoint_id','N/A')[:8]}...")
            print(f"  该检查点的消息数: {len(old_cp.values['messages'])}")

            old_config=old_cp.config
            r_travel=graph.invoke(
                {"messages":[HumanMessage(content="从过去重新开始")]},
                old_config
            )
            print(f"  时间旅行后 AI: {r_travel['messages'][-1].content}")

        print("\n" + "=" * 60)
        print("📊 最终状态")
        print("=" * 60)
        final_state=graph.get_state(config)
        for i,msg in enumerate(final_state.values["messages"]):
            role= "👤" if isinstance(msg,HumanMessage) else "🤖"
            print(f"  [{i}] {role} {msg.content[:50]}...")

if __name__ == "__main__":
    main()
