"""
记住用户偏好的 Agent
综合运用：短期记忆 + 长期记忆 + 记忆工具
"""
import os
import json
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
import threading

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ==================================================
# 1. 长期记忆存储
# ==================================================
MEMORY_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"user_preferences.json")
_prefs_lock=threading.Lock()

def load_preferences():
    with _prefs_lock:
        if not os.path.exists(MEMORY_FILE):
            return {}
        with open(MEMORY_FILE,"r",encoding="utf-8") as f:
            content=f.read().strip()
            if not content:
                return {}
            return json.loads(content)

def save_preferences(prefs):
    with _prefs_lock:
        with open(MEMORY_FILE,"w",encoding="utf-8") as f:
            json.dump(prefs,f,ensure_ascii=False,indent=2)

def get_user_prefs(user_id:str) -> dict:
    prefs=load_preferences()
    return prefs.get(user_id,{})

def update_pref(user_id:str,key:str,value:str):
    prefs=load_preferences()
    if user_id not in prefs:
        prefs[user_id]={}
    prefs[user_id][key]=value
    save_preferences(prefs)


# ==================================================
# 2. 记忆工具
# ==================================================
class SavePrefInput(BaseModel):
    user_id:str=Field(description="用户ID")
    key:str=Field(description="偏好类型，如'name'、'language'、'style'")
    value:str=Field(description="偏好值")

@tool(args_schema=SavePrefInput)
def save_preference(user_id:str,key:str,value:str) -> str:
    """保存用户偏好。
    
    当用户提到关于自己的信息时调用，比如：
    - 名字
    - 喜欢的编程语言
    - 代码风格偏好
    - 其他习惯
    
    Args:
        user_id: 用户ID
        key: 偏好类型
        value: 偏好值
    """
    update_pref(user_id,key,value)
    print(f"   💾 [保存偏好] {user_id}: {key} = {value}")
    return f"已记住：{key} = {value}"

class GetPrefInput(BaseModel):
    user_id:str=Field(description="用户ID")

@tool(args_schema=GetPrefInput)
def get_preferences(user_id:str) -> str:
    """获取用户的所有偏好。"""
    prefs=get_user_prefs(user_id)
    print(f"   📖 [读取偏好] {user_id}: {prefs}")
    return json.dumps(prefs,ensure_ascii=False)

PREFERENCE_TOOLS=[save_preference,get_preferences]


# ==================================================
# 3. 状态定义
# ==================================================
class AgentState(TypedDict):
    messages:Annotated[list,add_messages]
    user_id:str


# ==================================================
# 4. 定义节点
# ==================================================
model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

model_with_tools=model.bind_tools(PREFERENCE_TOOLS)

def agent_node(state:AgentState) -> dict:
    user_id=state.get("user_id","unknown")
    user_prefs=get_user_prefs(user_id)

    pref_text="\n".join([f"  - {k}: {v}" for k,v in user_prefs.items()]) if user_prefs else "  （暂无）"

    system_prompt=f"""你是一个贴心的助手。

当前用户ID: {user_id}
已知用户偏好：
{pref_text}

工作原则：
1. 当用户提到关于自己的信息（名字、偏好、习惯）时，调用 save_preference 工具保存
2. 回答时参考已知的用户偏好
3. 如果用户问"你记得我吗"，可以调用 get_preferences 查看

重要：用户ID 是 {user_id}，保存偏好时必须用这个ID。"""

    messages=[SystemMessage(content=system_prompt)]+state["messages"]

    print(f"\n[Agent] 消息数: {len(state['messages'])}, 已知偏好: {len(user_prefs)} 个")

    response=model_with_tools.invoke(messages)
    return {"messages":[response]}

def should_continue(state:AgentState) -> Literal["tools","__end__"]:
    last_message=state["messages"][-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

# ==================================================
# 5. 构建图
# ==================================================
builder=StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("tools",ToolNode(PREFERENCE_TOOLS))

builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",should_continue,{"tools":"tools","__end__":END})
builder.add_edge("tools","agent")

checkpointer=MemorySaver()
graph=builder.compile(checkpointer=checkpointer)


# ==================================================
# 6. 测试
# ==================================================
def chat(user_id:str,thread_id:str,message:str) -> str:
    config={"configurable":{"thread_id":thread_id}}
    result=graph.invoke(
        {"messages":[HumanMessage(content=message)],"user_id":user_id},
        config
    )
    return result["messages"][-1].content

def main():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    print("=" * 60)
    print("🧪 记住用户偏好的 Agent")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("📅 会话1（thread_id='session_1'）")
    print("=" * 60)

    print(f"\n用户: 你好，我叫张三，我是一名Python开发者，我喜欢简洁的代码")
    reply=chat("user_001","session_1","你好，我叫张三，我是一名Python开发者，我喜欢简洁的代码")
    print(f"AI: {reply}")

    print(f"\n用户: 我平时喜欢用函数式编程")
    reply=chat("user_001","session_1","我平时喜欢用函数式编程")
    print(f"AI: {reply}")

    print("\n" + "=" * 60)
    print("📅 会话2（thread_id='session_2'，模拟下次再来）")
    print("=" * 60)

    print(f"\n用户: 你还记得我吗？我叫什么名字？")
    reply=chat("user_001","session_2", "你还记得我吗？我叫什么名字？")
    print(f"AI: {reply}")

    print(f"\n用户: 我喜欢什么编程风格？")
    reply=chat("user_001","session_2","我喜欢什么编程风格？")
    print(f"AI: {reply}")

    print("\n" + "=" * 60)
    print("📊 存储的用户偏好")
    print("=" * 60)
    prefs=load_preferences()
    print(json.dumps(prefs,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
