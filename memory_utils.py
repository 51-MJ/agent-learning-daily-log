import json
import os

# 固定写绝对路径，彻底杜绝路径/拼写错误
HISTORY_PATH = r"D:\develop\agent-learing\agent-learning-daily-log\chat_history.json"

def load_chat_history():
    import json
    HISTORY_PATH = "chat_history.json"
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            full_history = json.load(f)
    except:
        return []
    # 滑动窗口核心：只取最后10条（5轮对话）
    short_history = full_history[-10:] if len(full_history) > 10 else full_history
    return short_history

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