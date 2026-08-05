"""
demo03 学习内容：Agent长期持久记忆开发
知识点：
1. while True只是会话内短期记忆，进程结束内存清空，对话丢失
2. 将每轮问答写入chat_history.json持久化保存到硬盘
练习目标：掌握短期记忆与长期记忆区别、对话落盘存储
"""
from smolagents import LiteLLMModel, CodeAgent, tool
import datetime
import json
import os

model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="dummy"
)

@tool
def get_full_datetime() -> str:
    """获取电脑当前完整时间：年月日、星期、时分秒，仅用户询问时间相关内容时调用"""
    week_map = {0:"星期一", 1:"星期二", 2:"星期三", 3:"星期四", 4:"星期五", 5:"星期六", 6:"星期日"}
    now = datetime.datetime.now()
    week = week_map[now.weekday()]
    date_part = f"{now.year}年{now.month}月{now.day}日 {week}"
    time_part = now.strftime("%H:%M:%S")
    return f"{date_part} {time_part}"

HISTORY_FILE = "chat_history.json"
chat_history = []
# 加载历史对话
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)

agent = CodeAgent(
    model=model,
    tools=[get_full_datetime],
    add_base_tools=False,
)

print("=== demo03：长期记忆智能体启动，输入exit退出 ===")
print(f"本次启动加载历史对话条数：{len(chat_history)}")

while True:
    user_msg = input("\n你：")
    if user_msg.lower() == "exit":
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        print("助手：对话已持久化保存，本次会话结束。")
        break

    reply = agent.run(user_msg)
    print(f"助手：{reply}")

    # 保存本轮对话
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": reply})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)