import json
import os

# 固定写绝对路径，彻底杜绝路径/拼写错误
HISTORY_PATH = r"D:\develop\agent-learing\agent-learning-daily-log\chat_history.json"

def load_chat_history():
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            full_history = json.load(f)
    except:
        return []
    return full_history

def append_chat_message(role: str, content: str):
    """
    追加一条对话并立刻保存到json
    :param role: "user" 或 "assistant"
    :param content: 对话内容文本
    """
    history = load_chat_history()
    history.append({
        "role": role,
        "content": content
    })
    # 写回文件
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_history_count():
    """获取历史对话总条数，方便启动打印"""
    return len(load_chat_history())

def slide_window_history(max_turn: int = 6) -> list:
     
     """
    滑动窗口截断对话历史
    :param max_turn: 最多保留多少轮问答，1轮=用户1条+助手1条
    :return: 截断后的最新历史列表
    """
     full_history = load_chat_history()
     max_save_items = max_turn * 2
     if len(full_history) > max_save_items:
        return full_history[-max_save_items:]
     return full_history

# ---------------------- 以下为Demo16新增：对话总结 ----------------------
import json as _json

SUMMARY_PATH = r"D:\develop\agent-learing\agent-learning-daily-log\summary.json"

def load_conversation_summary() -> str:
    """读取早期对话摘要，没有则返回空字符串"""
    try:
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
            return data.get("summary", "")
    except:
        return ""

def save_conversation_summary(summary: str):
    """保存对话摘要到文件"""
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        _json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

def build_summary_prompt(history_list: list) -> str:
    """
    组装用于生成摘要的prompt
    :param history_list: 需要被总结的早期对话列表
    :return: 摘要生成prompt
    """
    history_str = ""
    for item in history_list:
        history_str += f"{item['role']}：{item['content']}\n"

    return f"""
请将以下对话历史总结成一段简洁的摘要，保留关键信息（人物、事件、重要结论、用户核心诉求），不要遗漏重要细节，控制在200字以内。

【对话历史】
{history_str}

【摘要】
"""

__all__ = [
    "load_chat_history",
    "append_chat_message",
    "get_history_count",
    "slide_window_history",
    "load_conversation_summary",
    "save_conversation_summary",
    "build_summary_prompt"
]
