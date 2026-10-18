"""
LLM 摘要压缩示例
把早期对话压缩成摘要
"""
import os
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

class ChatState(TypedDict):
    messages:Annotated[list,add_messages]
    summary:str

model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def summarize_messages(messages:list) -> str:
    if not messages:
        return ""
    
    text="\n".join([
        f"{'用户' if isinstance(m,HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    ])

    prompt=f"""请把以下对话压缩成简洁的摘要，保留关键信息（用户说的名字、偏好、重要事实等）：

{text}

摘要（用项目符号列出关键点）："""

    response=model.invoke([HumanMessage(content=prompt)])
    return response.content

def call_model(state:ChatState) -> dict:
    messages=state["messages"]
    summary=state.get("summary","")

    MAX_MESSAGES=8
    KEEP_RECENT=4

    new_summary=summary

    if len(messages) > MAX_MESSAGES:
        old_messages=messages[:-KEEP_RECENT]
        recent_messages=messages[-KEEP_RECENT:]

        print(f"\n[压缩] 把 {len(old_messages)} 条旧消息压缩成摘要")

        combined_text=summary+"\n"+"\n".join([
            f"{'用户' if isinstance(m,HumanMessage) else 'AI'}: {m.content}"
            for m in old_messages
        ])

        new_summary=summarize_messages(old_messages)
        if summary:
            new_summary=f"{summary}\n{new_summary}"

        print(f"[新摘要] {new_summary[:100]}...")

        llm_messages=[SystemMessage(content=f"之前的对话摘要：\n{new_summary}")]+recent_messages
    else:
        llm_messages=messages

    print(f"[节点执行] 消息数：{len(llm_messages)}, 有摘要：{bool(new_summary)}")

    response=model.invoke(llm_messages)
    return {"messages":[response],"summary":new_summary}

builder=StateGraph(ChatState)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

checkpointer=MemorySaver()
graph=builder.compile(checkpointer=checkpointer)

def main():
    print("=" * 60)
    print("🧪 LLM 摘要压缩")
    print("=" * 60)

    config={"configurable":{"thread_id":"summary_demo"}}

    conversations=[
        "我叫张三，是一名Python开发者",
        "我最近在学习LangGraph",
        "我比较喜欢简洁的代码风格",
        "我养了一只猫叫小白",
        "我平时喜欢看科幻电影",
        "我住在北京",
        "我喜欢的编程语言是Python",
        "我不喜欢写文档",
        "我每天工作8小时",
        "我周末喜欢爬山",        
    ]

    for i,user_msg in enumerate(conversations,1):
        print(f"\n--- 第{i}轮 ---")
        print(f"用户：{user_msg}")

        result=graph.invoke({
            "messages":[HumanMessage(content=user_msg)],
            "summary":"",
        },config
        )

        response=result["messages"][-1]

        print(f"AI: {response.content[:60]}...")

    print("\n" + "=" * 60)
    print("📊 最终测试")
    print("=" * 60)

    result=graph.invoke({
        "messages":[HumanMessage(content="你还记得我叫什么名字吗？我养了什么宠物？")],
    },config
    )
    print(f"AI: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()
