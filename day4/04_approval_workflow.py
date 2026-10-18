"""
人工审批 Agent 完整流程
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.graph.message import add_messages
from typing import Annotated

class ApprovalState(TypedDict):
    messages:Annotated[list,add_messages]
    action:str
    action_params:dict
    decision:str
    status:str

def plan_action(state:ApprovalState) -> dict:
    print(f"🤔 Agent 规划中...")
    return {
        "action":"send_email",
        "action_params":{
            "to":"customer@example.com",
            "subject":"订单确认",
            "body":"您的订单已确认"
        },
        "status":"pending_approval"
    }

def human_review(state:ApprovalState) -> dict:
    action_info={
        "action":state["action"],
        "params":state["action_params"]
    }

    decision=interrupt({
        "message":"请审核以下操作",
        "action":action_info,
        "options":["approve","reject","edit"]
    })

    print(f"👤 审核决定: {decision}")
    return {"decision":decision}

def route_decision(state:ApprovalState) -> Literal["execute","reject","edit"]:
    if state["decision"]=="approve":
        return "execute"
    elif state["decision"]=="edit":
        return "edit"
    else:
        return "reject"

def execute_action(state:ApprovalState) -> dict:
    print(f"✅ 执行：{state['action']}")
    print(f"参数：{state['action_params']}")
    return {"status":"executed","messages":["动作已执行"]}

def reject_action(state:ApprovalState) -> dict:
    print(f"❌ 拒绝：{state['action']}")
    return {"status":"rejected","messages":["动作被拒绝"]}

def edit_action(state:ApprovalState) -> dict:
    print(f"✏️ 编辑后执行：{state['action']}")
    return {"status":"edited_executed","messages":["动作已修改并执行"]}

builder=StateGraph(ApprovalState)
builder.add_node("plan",plan_action)
builder.add_node("review",human_review)
builder.add_node("execute",execute_action)
builder.add_node("reject",reject_action)
builder.add_node("edit",edit_action)

builder.add_edge(START,"plan")
builder.add_edge("plan","review")

builder.add_conditional_edges(
    "review",
    route_decision,
    {
        "execute":"execute",
        "reject":"reject",
        "edit":"edit"
    }
)

builder.add_edge("execute",END)
builder.add_edge("reject",END)
builder.add_edge("edit",END)

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"approval-1"}}

print("=" * 60)
print("第1步：Agent 规划 → 中断等待审批")
print("=" * 60)

result=graph.invoke(
    {"messages":[],"action":"","action_params":{},"decision":"","status":""},
    config
)

state=graph.get_state(config)
print(f"\n⏸️ 中断在节点: {state.next}")
if state.tasks and state.tasks[0].interrupts:
    interrupt_value=state.tasks[0].interrupts[0].value
    print(f"📋 待审批操作:")
    print(f"   动作: {interrupt_value['action']['action']}")
    print(f"   参数: {interrupt_value['action']['params']}")
    print(f"   选项: {interrupt_value['options']}")

print("\n" + "=" * 60)
print("第2步：人类审批")
print("=" * 60)

human_decision="approve"
print(f"👤 人类选择: {human_decision}")

result=graph.invoke(Command(resume=human_decision),config)

print(f"\n📊 最终状态: {result['status']}")
print(f"📝 消息: {result['messages']}")