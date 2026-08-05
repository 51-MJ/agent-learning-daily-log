"""
demo04：路由分流 + 可真实回忆的长期记忆（Ollama原生接口版）
学习内容：
1. 拆分分工：普通闲聊/问答使用Ollama原生HTTP接口，工具调用继续使用smolagents
2. 关键词路由：通过关键词判断用户提问是否需要调用工具查询时间
3. 自主管理长期记忆：自己读写chat_history.json文件，启动加载历史、对话自动落盘
4. 解决smolagents原有痛点：内部记忆封闭、无法读取自定义对话json、重启后胡乱编造历史内容
练习目标：
- 掌握如何绕开smolagents记忆限制，自主掌控对话上下文
- 理解路由分流思想：简单问题快速回答，复杂带工具需求走Agent多步推理
- 验证重启程序后，模型可以根据本地json真实回忆过往聊天内容
"""


import requests
import json
import os
from smolagents import LiteLLMModel, CodeAgent, tool
import datetime

# ===================== 配置区 =====================
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"
# 对话长期记忆文件，放在项目根目录
HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.json")

# 初始化 smolagents 模型，只用来做工具调用
model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="dummy"
)

# 自定义时间查询工具
@tool
def get_full_datetime() -> str:
    """获取电脑当前完整时间：年月日、星期、时分秒"""
    week_map = {0:"星期一", 1:"星期二", 2:"星期三", 3:"星期四", 4:"星期五", 5:"星期六", 6:"星期日"}
    now = datetime.datetime.now()
    week = week_map[now.weekday()]
    date_part = f"{now.year}年{now.month}月{now.day}日 {week}"
    time_part = now.strftime("%H:%M:%S")
    return f"{date_part} {time_part}"

agent_tool = CodeAgent(
    model=model,
    tools=[get_full_datetime],
    add_base_tools=False
)

# ===================== 加载/保存长期记忆（自己掌控json） =====================
chat_history = []
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)

def save_history(user_msg: str, assistant_reply: str):
    """把本轮对话写入json持久化"""
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": assistant_reply})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def chat_normal(user_msg: str) -> str:
    """普通闲聊/问答：调用Ollama原生接口，带上全部历史"""
    messages = chat_history + [{"role": "user", "content": user_msg}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload)
    data = resp.json()
    answer = data["message"]["content"]
    save_history(user_msg, answer)
    return answer

def route_judge(user_msg: str) -> bool:
    """关键词路由：判断是否需要调用工具查时间"""
    time_keywords = ["几点","时间","日期","星期","几号","现在时刻"]
    for kw in time_keywords:
        if kw in user_msg:
            return True
    return False

# ===================== 主循环 =====================
if __name__ == "__main__":
    print("=== demo04 路由分流+可真实回忆长期记忆启动 ===")
    print(f"已加载历史对话 {len(chat_history)} 条，输入exit退出")
    while True:
        user_input = input("\n你：")
        if user_input.strip().lower() == "exit":
            print("助手：对话已保存，程序退出")
            break
        if route_judge(user_input):
            # 需要工具：走smolagents
            reply = agent_tool.run(user_input)
            save_history(user_input, reply)
        else:
            # 普通聊天：走原生Ollama接口，自带完整记忆
            reply = chat_normal(user_input)
        print(f"助手：{reply}")