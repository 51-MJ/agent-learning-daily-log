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

# 更新导出列表
__all__ = ["load_chat_history", "append_chat_message", "get_history_count", "slide_window_history"]