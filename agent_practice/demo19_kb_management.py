# demo19_kb_management.py
# 功能：Web界面 + 知识库管理，支持网页上传文件、查看知识库、删除文档
# 前置Python知识点：
#   1. gr.Blocks：Gradio高级布局，做多Tab页面
#   2. gr.Tab：选项卡，聊天和知识库管理分开
#   3. gr.File：文件上传组件
#   4. 按钮.click()：点击按钮时调用指定函数
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
from rag_utils import (
    search_knowledge,
    save_turn_to_history,
    search_history,
    add_knowledge,
    delete_knowledge_by_source,
    list_knowledge_sources
)

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


def build_prompt(user_input, summary, short_history, tool_info, rag_text, history_recall):
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
    check_and_do_summary()
    summary = load_conversation_summary()
    short_history = slide_window_history(max_turn=WINDOW_TURN)
    intent = classify_intent(user_input)
    tool_info = run_tools(user_input, intent)
    rag_text = search_knowledge(user_input, top_n=3)
    history_recall = search_history(user_input, top_n=3)

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
                yield full_answer
            if data.get("done", False):
                break

    append_chat_message("user", user_input)
    append_chat_message("assistant", full_answer)
    save_turn_to_history(user_input, full_answer)


# ---------------------- 知识库管理相关函数 ----------------------
def refresh_kb_list():
    """刷新知识库来源列表，返回显示文本"""
    sources = list_knowledge_sources()
    if not sources:
        return "知识库为空，暂无文档"
    lines = ["当前知识库包含以下文档："]
    for s in sources:
        lines.append(f"- {s}")
    return "\n".join(lines)


def upload_kb_file(file_obj):
    """处理上传的txt文件，自动入库"""
    if file_obj is None:
        return "请先选择要上传的txt文件", refresh_kb_list()
    # file_obj是Gradio上传的文件对象，有.name属性
    file_name = Path(file_obj.name).name
    # 读取文件内容
    with open(file_obj.name, "r", encoding="utf-8") as f:
        content = f.read()
    # 调用add_knowledge入库
    add_knowledge([content], source=file_name)
    return f"✅ 已上传并入库：{file_name}", refresh_kb_list()


def delete_kb_source(source_name):
    """删除指定来源的知识库"""
    if not source_name.strip():
        return "请输入要删除的文件名", refresh_kb_list()
    delete_knowledge_by_source(source_name.strip())
    return f"✅ 已删除：{source_name}", refresh_kb_list()


# ---------------------- 构建Web界面 ----------------------
with gr.Blocks(title="本地RAG智能体") as demo:
    gr.Markdown("# 本地RAG智能体")

    with gr.Tab("聊天"):
        gr.ChatInterface(
            fn=chat_response,
            title="对话窗口",
            description="支持知识库检索、历史对话召回、对话总结"
        )

    with gr.Tab("知识库管理"):
        gr.Markdown("## 知识库管理")

        # 显示当前知识库列表
        kb_display = gr.Textbox(
            label="当前知识库文档",
            value=refresh_kb_list(),
            lines=8,
            interactive=False
        )

        # 上传文件区
        gr.Markdown("### 上传新文档（仅支持.txt）")
        upload_file = gr.File(label="选择txt文件")
        upload_btn = gr.Button("上传并入库", variant="primary")
        upload_status = gr.Textbox(label="上传状态", interactive=False)

        # 删除文档区
        gr.Markdown("### 删除文档")
        delete_input = gr.Textbox(label="要删除的文件名（如：people.txt）")
        delete_btn = gr.Button("删除", variant="stop")
        delete_status = gr.Textbox(label="删除状态", interactive=False)

        # 刷新按钮
        refresh_btn = gr.Button("刷新知识库列表")

        # 绑定按钮事件
        upload_btn.click(
            upload_kb_file,
            inputs=[upload_file],
            outputs=[upload_status, kb_display]
        )
        delete_btn.click(
            delete_kb_source,
            inputs=[delete_input],
            outputs=[delete_status, kb_display]
        )
        refresh_btn.click(
            refresh_kb_list,
            outputs=[kb_display]
        )


if __name__ == "__main__":
    demo.launch()
