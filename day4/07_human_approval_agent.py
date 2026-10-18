"""
人工审核 Agent：高风险工具自动中断审批
"""
import os
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from typing import Annotated

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ============ 定义工具 ============
@tool
def query_data(table:str) -> str:
    """查询表数据（低风险，自动执行）"""
    return f"查询表 {table} 的数据：共100条记录"

@tool
def delete_table(table:str) -> str:
    """删除数据表（高风险，需要审批）"""
    return f"已删除 {table}"

@tool
def delete_file(filename:str) -> str:
    """删除文件（高风险，需要审批）"""
    return f"已删除文件 {filename}"


# ============ 需要审批的工具列表 ============
SENSITIVE_TOOLS=["delete_table","delete_file"]


# ============ State 定义 ============
class AgentState(TypedDict):
    messages:Annotated[list,add_messages]
    pending_action:dict
    approval_decision:str


# ============ 模型 ============
model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

tools=[query_data,delete_table,delete_file]
model_with_tools=model.bind_tools(tools)


# ============ 节点：Agent 思考 ============
def agent_node(state:AgentState) -> dict:
    messages=state["messages"]
    system_msg=SystemMessage(content="""你是一个数据库管理助手。
可用工具：
- query_data: 查询数据（安全）
- delete_table: 删除表（危险，需要审批）
- delete_file: 删除文件（危险，需要审批）

根据用户需求选择合适的工具。""")

    response=model_with_tools.invoke([system_msg]+messages)
    return {"messages":[response]}


# ============ 节点：人工审批 ============
def approval_node(state:AgentState) -> dict:
    last_message=state["messages"][-1]

    tool_calls=last_message.tool_calls
    if not tool_calls:
        return {"approval_decision":"no_tool_call"}

    pending=[tc for tc in tool_calls if tc["name"] in SENSITIVE_TOOLS]

    if not pending:
        return {"approval_decision":"auto_approve"}

    action_info=[]
    for tc in pending:
        action_info.append({
            "tool":tc["name"],
            "params":tc["args"]
        })

    decision=interrupt({
        "message":"以下操作需要人工审批",
        "actions":action_info,
        "options":["approve","reject"]
    })

    print(f"👤 审批决定：{decision}")
    return {"approval_decision":decision}


# ============ 路由函数 ============
def should_continue(state:AgentState) -> Literal["approval","tools","__end__"]:
    last_message=state["messages"][-1]

    if not hasattr(last_message,"tool_calls") or not last_message.tool_calls:
        return "__end__"

    has_sensitive=any(
        tc["name"] in SENSITIVE_TOOLS
        for tc in last_message.tool_calls
    )

    if has_sensitive:
        return "approval"
    return "tools"

def route_after_approval(state:AgentState) -> Literal["tools","__end__"]:
    if state["approval_decision"]=="approve":
        return "tools"
    elif state["approval_decision"]=="auto_approve":
        return "tools"
    else:
        return "__end__"

    
# ============ 构建图 ============
builder=StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("approval",approval_node)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "approval":"approval",
        "tools":"tools",
        "__end__":END
    }
)
builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {"tools":"tools","__end__":END}
)
builder.add_edge("tools","agent")

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)


# ============ 测试 ============
def run_with_approval(user_input:str,thread_id:str,auto_approve:bool=False):
    config={"configurable":{"thread_id":thread_id}}

    print(f"\n{'='*60}")
    print(f"👤 用户: {user_input}")
    print(f"{'='*60}")

    result=graph.invoke(
        {"messages":[HumanMessage(content=user_input)],
         "pending_action":{},
         "approval_decision":""},
         config
    )

    state=graph.get_state(config)

    while state.next:
        print(f"\n⏸️ 中断在: {state.next}")

        if state.tasks and state.tasks[0].interrupts:
            interrupt_data=state.tasks[0].interrupts[0].value
            print(f"\n📋 待审批操作:")
            for action in interrupt_data.get("actions",[]):
                print(f"    工具：{action['tool']}")
                print(f"    参数：{action['params']}")

        if auto_approve:
            decision="approve" 
        else:
            decision=input("\n👤 审批决定（approve/reject）：").strip()
            if decision not in ["approve","reject"]:
                decision="reject"

        print(f"👤 决定：{decision}")

        result=graph.invoke(Command(resume=decision),config)
        state=graph.get_state(config)

    final_message=result["messages"][-1]
    print(f"\n🤖 Agent: {final_message.content}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 人工审核 Agent 测试")
    print("=" * 60)    

    run_with_approval("查询 product 表的数据","test-1",auto_approve=True)

    run_with_approval("删除 user 表","test-2",auto_approve=True)

    run_with_approval("删除 config.txt 文件","test-3",auto_approve=False)