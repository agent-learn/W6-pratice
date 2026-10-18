"""
长期记忆存储模块
用 JSON 文件存储用户信息
"""
import os
import json
from typing import Dict, Any

MEMORY_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"long_term_memory.json")

def load_memory() -> Dict[str,Any]:
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory:Dict[str,Any]):
    with open(MEMORY_FILE,"w",encoding="utf-8") as f:
        json.dump(memory,f,ensure_ascii=False,indent=2)

def get_user_memory(user_id:str) -> Dict[str,Any]:
    memory=load_memory()
    return memory.get(user_id,{})

def update_user_memory(user_id:str,key:str,value:Any):
    memory=load_memory()
    if user_id not in memory:
        memory[user_id]={}
    memory[user_id][key]=value
    save_memory(memory)

def get_memory_summary(user_id:str) -> str:
    user_memory=get_user_memory(user_id)
    if not user_memory:
        return "暂无用户信息。"

    lines=["已知用户信息："]
    for key,value in user_memory.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)

if __name__ == "__main__":
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    update_user_memory("user_001","name","张三")
    update_user_memory("user_001","language","Python")
    update_user_memory("user_002","name","李四")

    print(get_memory_summary("user_001"))
    print()
    print(get_memory_summary("user_002"))
