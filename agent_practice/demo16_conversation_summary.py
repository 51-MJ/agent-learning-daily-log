# demo16_conversation_summary.py
# 功能：对话总结 + 滑动窗口，早期对话自动生成摘要，解决长对话早期信息丢失问题
# 前置Python知识点：
#   1. 函数调用传参：把模型实例传给总结函数，让总结函数也能调用模型生成摘要
#   2. 阈值判断：对话总数超过阈值才触发总结，不是每轮都总结，节省资源
#   3. 字符串拼接：摘要文本和最近N轮对话文本拼接成完整上下文
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
from tool_router import classify_intent, run_tools
from rag_utils import search_knowledge

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

# 总结触发阈值：对话超过这么多条就触发一次总结
SUMMARY_TRIGGER = 12
# 滑动窗口保留最近多少轮
WINDOW_TURN = 6


def stream_chat(system_prompt: str, user_content: str) -> str:
    """流式调用模型，逐字打印，返回完整回答"""
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
    return full_answer


def generate_summary(history_list: list) -> str:
    """
    调用模型生成对话摘要（非流式，直接返回摘要文本）
    :param history_list: 需要被总结的早期对话列表
    :return: 摘要文本
    """
    prompt = build_summary_prompt(history_list)
    messages = [{"role": "user", "content": prompt}]
    payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def check_and_do_summary():
    """
    检查对话数量，超过阈值则触发总结
    逻辑：总对话数 > 阈值时，把滑动窗口之外的早期对话生成摘要
    """
    full_history = load_chat_history()
    if len(full_history) <= SUMMARY_TRIGGER:
        return  # 没超过阈值，不总结

    # 滑动窗口保留最近 WINDOW_TURN*2 条，前面的都需要被总结
    window_items = WINDOW_TURN * 2
    old_history = full_history[:-window_items]  # 早期对话（需要被总结的部分）

    if not old_history:
        return

    print(f"\n[对话已达{len(full_history)}条，自动总结早期对话...]")
    old_summary = load_conversation_summary()

    # 如果已有旧摘要，把旧摘要也加进去，让模型在旧摘要基础上更新
    if old_summary:
        history_with_old_summary = [{"role": "system", "content": f"已有摘要：{old_summary}"}] + old_history
        new_summary = generate_summary(history_with_old_summary)
    else:
        new_summary = generate_summary(old_history)

    save_conversation_summary(new_summary)
    print(f"[早期对话总结完成，摘要长度：{len(new_summary)}字]\n")


def build_prompt(user_input: str, summary: str, short_history: list, tool_info: str, rag_text: str) -> tuple:
    """组装最终prompt，包含早期摘要 + 最近对话 + 工具结果 + 知识库"""
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    # 有摘要就显示，没有就不显示
    summary_part = f"【早期对话摘要】\n{summary}\n\n" if summary else ""

    system_prompt = """你是一个知识库问答助手。
回答规则：
1. 优先基于知识库内容回答，引用知识库原文。
2. 允许基于知识库做合理逻辑推理。
3. 知识库没有的具体信息，明确说"知识库中没有相关信息"。
4. 回答简洁自然，不要复述规则。"""

    user_content = f"""
{summary_part}
【近期对话记录】
{history_str}

【工具查询结果】
{tool_info if tool_info else "无"}

【本地知识库检索内容】
{rag_text}

【用户提问】
{user_input}
"""
    return system_prompt, user_content


def main():
    print("=== Demo16 对话总结 + 滑动窗口智能体 ===")
    print(f"总结触发阈值：{SUMMARY_TRIGGER}条对话，滑动窗口：最近{WINDOW_TURN}轮")
    print("输入exit退出对话\n")

    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break

        # 第一步：检查是否需要总结早期对话
        check_and_do_summary()

        # 第二步：读取摘要 + 滑动窗口
        summary = load_conversation_summary()
        short_history = slide_window_history(max_turn=WINDOW_TURN)

        # 第三步：意图分类 + 工具执行
        intent = classify_intent(user_input)
        intent_names = {"time": "时间查询", "calculate": "数学计算", "search": "联网搜索", "knowledge": "知识库问答"}
        print(f"[意图识别：{intent_names.get(intent, intent)}]")
        tool_info = run_tools(user_input, intent)

        # 第四步：RAG检索
        if intent == "knowledge":
            rag_text = search_knowledge(user_input, top_n=3)
            if rag_text == "暂无相关知识库内容":
                print("[知识库无匹配，切换闲聊模式]")
                system_prompt = "你是一个友好的AI助手，结合常识自然回答用户问题，简洁友好。"
                user_content = f"【用户提问】{user_input}"
            else:
                system_prompt, user_content = build_prompt(user_input, summary, short_history, tool_info, rag_text)
        else:
            system_prompt = "你是一个友好的AI助手，结合工具查询结果和常识自然回答用户问题，简洁友好。"
            user_content = f"【工具查询结果】{tool_info}\n\n【用户提问】{user_input}"

        # 第五步：流式调用模型
        response = stream_chat(system_prompt, user_content)
        print()

        # 第六步：保存对话
        append_chat_message("user", user_input)
        append_chat_message("assistant", response)


if __name__ == "__main__":
    main()
