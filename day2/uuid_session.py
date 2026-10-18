"""
用 UUID 生成唯一 thread_id
"""
import uuid

thread_id=str(uuid.uuid4())
print(f"生成的 thread_id: {thread_id}")

config = {"configurable": {"thread_id": thread_id}}
