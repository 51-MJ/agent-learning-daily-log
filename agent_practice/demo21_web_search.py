# demo21_web_search.py
# 功能：联网搜索，Agent可以调用web_search工具获取互联网实时信息
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import json
from memory_utils import (
    load_chat_history,
    append_chat_message,
    slide_window_history,
    load_conversation_summary,
    save_conversation_summary,
    build_summary_prompt
)
from tool_router import run_tools, web_search
from rag_utils import search_knowledge, save_turn_to_history, search_history

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"
SUMMARY_TRIGGER = 12
WINDOW_TURN = 6


def generate_summary(history_list: list) -> str:
    prompt = build_summary_prompt(history_list)
    messages = [{"role": "user", "content": prompt}]
    payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def check_and_do_summary():
    full_history = load_chat_history()
    if len(full_history) <= SUMMARY_TRIGGER:
        return
    window_items = WINDOW_TURN * 2
    old_history = full_history[:-window_items]
    if not old_history:
        return
    old_summary = load_conversation_summary()
    if old_summary:
        history_with_old_summary = [{"role": "system", "content": f"已有摘要：{old_summary}"}] + old_history
        new_summary = generate_summary(history_with_old_summary)
    else:
        new_summary = generate_summary(old_history)
    save_conversation_summary(new_summary)


def llm_classify_intent(user_input: str, history: list, summary: str) -> str:
    history_str = ""
    for item in history:
        history_str += f"{item['role']}：{item['content']}\n"

    prompt = f"""请判断以下用户提问属于哪种意图，只返回意图名称，不要输出其他内容：

意图选项：
- time：问时间、日期、星期
- calculate：问数学计算
- search：联网搜索最新资讯、实时信息
- knowledge：问本地知识、事实、概念
- chat：闲聊、打招呼、随便聊聊

【对话历史】
{history_str}

【用户提问】
{user_input}

只返回一个意图名称："""

    messages = [{"role": "user", "content": prompt}]
    payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    intent = data.get("message", {}).get("content", "").strip().lower()

    if "time" in intent:
        return "time"
    elif "calculate" in intent:
        return "calculate"
    elif "search" in intent:
        return "search"
    elif "knowledge" in intent:
        return "knowledge"
    else:
        return "chat"


def main():
    print("=== Demo21 联网搜索智能体 ===")
    print("输入exit退出\n")

    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break

        check_and_do_summary()
        summary = load_conversation_summary()
        short_history = slide_window_history(max_turn=WINDOW_TURN)
        history_recall = search_history(user_input, top_n=3)

        # 模型判断意图
        intent = llm_classify_intent(user_input, short_history, summary)
        print(f"[模型判断意图：{intent}]")

        # 根据意图执行对应流程
        if intent == "time":
            tool_info = run_tools(user_input, intent)
            system_prompt = "你是一个友好的AI助手，根据时间信息简洁回答用户问题。"
            user_content = f"【时间信息】{tool_info}\n\n【用户提问】{user_input}"

        elif intent == "calculate":
            tool_info = run_tools(user_input, intent)
            system_prompt = "你是一个友好的AI助手，根据计算结果简洁回答用户问题。"
            user_content = f"【计算结果】{tool_info}\n\n【用户提问】{user_input}"

        elif intent == "search":
            # 联网搜索
            print("[正在联网搜索...]")
            search_result = web_search(user_input, max_results=3)
            print(f"[搜索完成]")
            system_prompt = """你是一个AI助手，请根据联网搜索结果回答用户问题。
如果搜索结果和问题相关，基于搜索结果回答。
如果搜索结果不相关，用你自己的知识回答。
回答简洁自然。"""
            user_content = f"""【联网搜索结果】
{search_result}

【用户提问】{user_input}"""

        elif intent == "knowledge":
            rag_text = search_knowledge(user_input, top_n=3)
            if rag_text == "暂无相关知识库内容":
                print("[知识库无匹配，切换闲聊模式]")
                system_prompt = "你是一个友好的AI助手，用常识自然回答用户问题。"
                user_content = f"【相关历史】\n{history_recall}\n\n【用户提问】{user_input}"
            else:
                system_prompt = "你是一个知识库问答助手，基于知识库内容回答用户问题。"
                user_content = f"""【知识库内容】\n{rag_text}\n\n【用户提问】{user_input}"""

        else:
            # chat闲聊
            system_prompt = "你是一个友好的AI助手，像朋友一样自然回答用户问题。"
            user_content = f"【相关历史】\n{history_recall}\n\n【用户提问】{user_input}"

        # 流式调用模型
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        payload = {"model": MODEL_NAME, "messages": messages, "stream": True}
        response = requests.post(OLLAMA_URL, json=payload, stream=True)

        full_answer = ""
        print("Agent：", end="", flush=True)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data:
                    content = data["message"].get("content", "")
                    print(content, end="", flush=True)
                    full_answer += content
                if data.get("done", False):
                    break
        print()

        # 保存对话
        append_chat_message("user", user_input)
        append_chat_message("assistant", full_answer)
        save_turn_to_history(user_input, full_answer)


if __name__ == "__main__":
    main()
