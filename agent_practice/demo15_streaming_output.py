# demo15_streaming_output.py
# 功能：流式输出 + 意图分类 + RAG检索，模型回答逐字打印，提升使用体验
# 前置Python知识点：
#   1. requests流式响应：stream=True + iter_lines() 逐行读取，不一次性加载全部响应
#   2. JSON逐行解析：流式接口每行返回一个JSON，用json.loads逐行解析
#   3. print的flush参数：flush=True强制立即输出，不缓冲，实现逐字显示效果
import sys
import json
import requests
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from memory_utils import append_chat_message, slide_window_history
from tool_router import classify_intent, run_tools
from rag_utils import search_knowledge

# Ollama服务地址和模型名（直接调用底层API实现流式，不经过smolagents的CodeAgent）
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"


def stream_chat(system_prompt: str, user_content: str) -> str:
    """
    调用Ollama流式API，逐字打印回答，返回完整回答文本
    :param system_prompt: 系统提示词（规则约束）
    :param user_content: 用户侧内容（包含问题、检索结果、工具结果等）
    :return: 模型生成的完整回答
    """
    # 组装messages，ollama的chat接口格式
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True   # 关键：开启流式输出
    }

    # 发送请求，stream=True表示不一次性读取响应
    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    full_answer = ""
    print("Agent：", end="", flush=True)  # 先打印前缀，不换行

    # iter_lines()逐行读取响应流，每行是一个JSON
    for line in response.iter_lines():
        if line:  # 跳过空行
            data = json.loads(line)  # 解析这一行的JSON
            # 流式返回的内容在message.content里
            if "message" in data:
                content = data["message"].get("content", "")
                # end=""不换行，flush=True强制立即输出，实现逐字显示
                print(content, end="", flush=True)
                full_answer += content
            # done=True表示生成结束
            if data.get("done", False):
                break

    print()  # 回答结束，换一行
    return full_answer


def build_knowledge_prompt(user_input: str, short_history: list, tool_info: str, rag_text: str) -> tuple:
    """
    知识库问答模式：返回(system_prompt, user_content)
    """
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    system_prompt = """你是一个知识库问答助手。
回答规则：
1. 优先基于知识库内容回答，引用知识库原文。
2. 允许基于知识库做合理逻辑推理（如初中生→学生）。
3. 知识库没有的具体信息（年龄、地址等），明确说"知识库中没有相关信息"。
4. 回答简洁自然，不要复述规则。"""

    user_content = f"""
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


def build_chat_prompt(user_input: str, short_history: list, tool_info: str) -> tuple:
    """
    闲聊模式：返回(system_prompt, user_content)
    """
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    system_prompt = """你是一个友好的AI助手。
回答规则：
1. 结合工具查询结果和你的常识，自然地回答用户问题。
2. 如果工具结果中有相关信息，优先参考工具结果。
3. 回答简洁友好，像正常聊天一样。"""

    user_content = f"""
【近期对话记录】
{history_str}

【工具查询结果】
{tool_info if tool_info else "无"}

【用户提问】
{user_input}
"""
    return system_prompt, user_content


def main():
    print("=== Demo15 流式输出 + 意图分类 + RAG智能体 ===")
    print("模型回答会逐字显示，输入exit退出对话\n")

    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break

        # 第一步：意图分类
        intent = classify_intent(user_input)
        intent_names = {
            "time": "时间查询",
            "calculate": "数学计算",
            "search": "联网搜索",
            "knowledge": "知识库问答"
        }
        print(f"[意图识别：{intent_names.get(intent, intent)}]")

        # 第二步：执行对应工具
        tool_info = run_tools(user_input, intent)

        # 第三步：滑动窗口取历史
        short_history = slide_window_history(max_turn=6)

        # 第四步：根据意图选择prompt模式
        if intent == "knowledge":
            rag_text = search_knowledge(user_input, top_n=3)
            if rag_text == "暂无相关知识库内容":
                # 检索为空 → 转闲聊模式
                print("[知识库无匹配，切换闲聊模式]")
                system_prompt, user_content = build_chat_prompt(user_input, short_history, tool_info)
            else:
                system_prompt, user_content = build_knowledge_prompt(user_input, short_history, tool_info, rag_text)
        else:
            # 工具类意图走闲聊模式
            system_prompt, user_content = build_chat_prompt(user_input, short_history, tool_info)

        # 第五步：流式调用模型，逐字打印
        response = stream_chat(system_prompt, user_content)
        print()  # 多空一行，美观

        # 第六步：保存对话
        append_chat_message("user", user_input)
        append_chat_message("assistant", response)


if __name__ == "__main__":
    main()
