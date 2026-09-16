
# demo17_history_recall.py
# 功能：相关历史召回，把历史对话存进向量库，提问时精准检索相关历史，解决长对话早期信息丢失问题
# 前置Python知识点：
#   1. 双路检索：同时检索知识库和历史对话库，结果合并进prompt
#   2. 边聊边存：每轮对话结束后，把当前轮存进历史向量库
#   3. 多Collection：知识库和历史对话存在不同collection，互不干扰
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
from rag_utils import search_knowledge, save_turn_to_history, search_history

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

SUMMARY_TRIGGER = 12
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
    """调用模型生成对话摘要（非流式）"""
    prompt = build_summary_prompt(history_list)
    messages = [{"role": "user", "content": prompt}]
    payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def check_and_do_summary():
    """检查对话数量，超过阈值则触发总结"""
    full_history = load_chat_history()
    if len(full_history) <= SUMMARY_TRIGGER:
        return

    window_items = WINDOW_TURN * 2
    old_history = full_history[:-window_items]
    if not old_history:
        return

    print(f"\n[对话已达{len(full_history)}条，自动总结早期对话...]")
    old_summary = load_conversation_summary()

    if old_summary:
        history_with_old_summary = [{"role": "system", "content": f"已有摘要：{old_summary}"}] + old_history
        new_summary = generate_summary(history_with_old_summary)
    else:
        new_summary = generate_summary(old_history)

    save_conversation_summary(new_summary)
    print(f"[早期对话总结完成]\n")


def build_prompt(user_input: str, summary: str, short_history: list,
                 tool_info: str, rag_text: str, history_recall: str) -> tuple:
    """组装完整prompt：摘要 + 滑动窗口 + 相关历史召回 + 工具结果 + 知识库"""
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    summary_part = f"【早期对话摘要】\n{summary}\n\n" if summary else ""
    history_recall_part = f"{history_recall}\n\n" if history_recall else ""

    system_prompt = """你是一个知识库问答助手。
回答规则：
1. 优先基于知识库内容回答，引用知识库原文。
2. 允许基于知识库做合理逻辑推理。
3. 知识库没有的具体信息，明确说"知识库中没有相关信息"。
4. 结合历史对话参考回答用户的指代性问题。
5. 回答简洁自然，不要复述规则。"""

    user_content = f"""
{summary_part}
{history_recall_part}
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
    print("=== Demo17 相关历史召回智能体 ===")
    print(f"总结触发：{SUMMARY_TRIGGER}条，滑动窗口：最近{WINDOW_TURN}轮")
    print("输入exit退出对话\n")

    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break

        # 第一步：检查是否需要总结
        check_and_do_summary()

        # 第二步：读取摘要 + 滑动窗口
        summary = load_conversation_summary()
        short_history = slide_window_history(max_turn=WINDOW_TURN)

        # 第三步：意图分类 + 工具执行
        intent = classify_intent(user_input)
        intent_names = {"time": "时间查询", "calculate": "数学计算", "search": "联网搜索", "knowledge": "知识库问答"}
        print(f"[意图识别：{intent_names.get(intent, intent)}]")
        tool_info = run_tools(user_input, intent)

        # 第四步：双路检索（知识库 + 历史对话）
        rag_text = search_knowledge(user_input, top_n=3)
        print(f"[RAG检索结果] {rag_text[:100]}")  # 打印前100字看看
        history_recall = search_history(user_input, top_n=3)

        # 第五步：组装prompt
        if rag_text == "暂无相关知识库内容" and intent == "knowledge":
            print("[知识库无匹配，切换闲聊模式]")
            system_prompt = """你是一个友好的AI助手。
                            现在是闲聊模式，知识库中没有相关信息。
                            请用你自己的常识和知识自然回答用户问题。
                            不要说"知识库中没有相关信息"这种话，直接用你的知识回答即可。
                            回答简洁自然，像和朋友聊天一样。"""
            user_content = f"【相关历史对话参考】\n{history_recall}\n\n【用户提问】{user_input}"
        else:
            system_prompt, user_content = build_prompt(
                user_input, summary, short_history, tool_info, rag_text, history_recall
            )

        # 第六步：流式调用模型
        response = stream_chat(system_prompt, user_content)
        print()

        # 第七步：保存对话 + 存进历史向量库
        append_chat_message("user", user_input)
        append_chat_message("assistant", response)
        save_turn_to_history(user_input, response)


if __name__ == "__main__":
    main()
