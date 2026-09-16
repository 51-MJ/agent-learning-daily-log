# demo18_web_ui.py
# 功能：Web聊天界面，把命令行程序变成可视化网页，支持流式输出
# 前置Python知识点：
#   1. Gradio库：几行代码做Web界面，不用写前端
#   2. yield关键字：生成器函数，每次返回一点结果，实现流式输出
#   3. 函数当参数传：Gradio把我们的函数绑定到界面上，用户点发送时自动调用
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import json
import gradio as gr
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
    print(f"[对话已达{len(full_history)}条，自动总结早期对话...]")
    old_summary = load_conversation_summary()
    if old_summary:
        history_with_old_summary = [{"role": "system", "content": f"已有摘要：{old_summary}"}] + old_history
        new_summary = generate_summary(history_with_old_summary)
    else:
        new_summary = generate_summary(old_history)
    save_conversation_summary(new_summary)
    print("[早期对话总结完成]")


def build_prompt(user_input: str, summary: str, short_history: list,
                 tool_info: str, rag_text: str, history_recall: str):
    """组装完整prompt"""
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"
    summary_part = f"【早期对话摘要】\n{summary}\n\n" if summary else ""
    history_recall_part = f"{history_recall}\n\n" if history_recall else ""

    system_prompt = """你是一个知识库问答助手。
回答规则：
1. 如果知识库内容和问题明显相关，基于知识库回答。
2. 如果知识库内容和问题不相关，或者问题是闲聊类，用你自己的常识自然回答。
3. 不要一上来就说"知识库中没有相关信息"，先判断检索内容和问题有没有关系。
4. 回答简洁自然，像和朋友聊天一样。"""

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


def chat_response(user_input, chat_history):
    """
    Gradio调用的聊天函数
    :param user_input: 用户输入的文字
    :param chat_history: Gradio维护的对话历史（列表）
    :yield: 逐字返回回答，实现流式效果
    """
    # 1. 检查是否需要总结
    check_and_do_summary()

    # 2. 读取摘要 + 滑动窗口
    summary = load_conversation_summary()
    short_history = slide_window_history(max_turn=WINDOW_TURN)

    # 3. 意图分类 + 工具执行
    intent = classify_intent(user_input)
    tool_info = run_tools(user_input, intent)

    # 4. 双路检索
    rag_text = search_knowledge(user_input, top_n=3)
    history_recall = search_history(user_input, top_n=3)

    # 5. 组装prompt
    if rag_text == "暂无相关知识库内容" and intent == "knowledge":
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

    # 6. 流式调用模型，逐字yield给Gradio
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    payload = {"model": MODEL_NAME, "messages": messages, "stream": True}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    full_answer = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "message" in data:
                content = data["message"].get("content", "")
                full_answer += content
                yield full_answer  # 每次返回当前完整内容，Gradio自动逐字显示
            if data.get("done", False):
                break

    # 7. 保存对话 + 存进历史向量库
    print("[DEBUG] 准备保存对话...")
    append_chat_message("user", user_input)
    append_chat_message("assistant", full_answer)
    save_turn_to_history(user_input, full_answer)
    print("[DEBUG] 对话保存完成")

# 启动Web界面
demo = gr.ChatInterface(
    fn=chat_response,
    title="本地RAG智能体",
    description="支持知识库检索、历史对话召回、对话总结的本地AI助手",
    
)

if __name__ == "__main__":
    demo.launch()
