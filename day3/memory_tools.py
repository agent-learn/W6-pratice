"""
记忆相关工具
让 Agent 可以主动保存用户信息
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from long_term_memory import update_user_memory, get_user_memory

class SaveMemoryInput(BaseModel):
    user_id:str=Field(description="用户ID")
    key:str=Field(description="记忆的键，如'name'、'language'、'preference'")
    value:str=Field(description="记忆的值")

@tool(args_schema=SaveMemoryInput)
def save_user_memory(user_id:str,key:str,value:str) -> str:
    """保存用户的长期记忆。
    
    使用场景：当用户提到关于自己的信息时（如名字、偏好、习惯），
    调用此工具保存，以便下次对话时记住。
    
    Args:
        user_id: 用户ID
        key: 记忆的键
        value: 记忆的值
    
    Returns:
        保存结果
    """    
    update_user_memory(user_id,key,value)
    return f"已保存：{key} = {value}"

class GetMemoryInput(BaseModel):
    user_id:str=Field(description="用户ID")

@tool(args_schema=GetMemoryInput)
def get_user_memory_tool(user_id:str) -> str:
    """获取用户的长期记忆。
    
    使用场景：当需要了解用户信息时调用。
    
    Args:
        user_id: 用户ID
    
    Returns:
        用户记忆的 JSON 字符串
    """    
    import json
    memory=get_user_memory(user_id)
    return json.dumps(memory,ensure_ascii=False)

MEMORY_TOOLS=[save_user_memory,get_user_memory_tool]

