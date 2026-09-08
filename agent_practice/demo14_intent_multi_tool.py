# demo14_intent_multi_tool.py
# 功能：意图分类架构 + 多工具扩展（时间/计算器/联网搜索）+ 知识库问答
# 前置Python知识点：
#   1. 函数多分支返回：根据不同条件返回不同结果
#   2. 字符串替换replace()：清洗用户输入，提取关键词
#   3. 集合set：用于字符白名单校验，防止eval注入
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from smolagents import CodeAgent, LiteLLMModel
from memory_utils import append_chat_message, slide_window_history
from tool_router import classify_intent, run_tools
from rag_utils import search_knowledge, list_knowledge_sources

MODEL_CONFIG = {
    "model_name": "ollama/qwen2.5:7b",
    "temperature": 0.7,
    "max_tokens": 1024
}
model = LiteLLMModel(
    model_id=MODEL_CONFIG["model_name"],
    temperature=MODEL_CONFIG["temperature"],
    max_tokens=MODEL_CONFIG["max_tokens"]
)
agent = CodeAgent(model=model, tools=[])


def build_knowledge_prompt(user_input: str, short_history: list, tool_info: str, rag_text: str) -> str:
    """知识库问答模式：严格约束，基于知识库回答"""
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    return f"""
【用户提问】{user_input}

【近期对话记录】
{history_str}

【工具查询结果】
{tool_info if tool_info else "无"}

【本地知识库检索内容】
{rag_text}

【回答规则】
1. 优先基于知识库内容回答，引用知识库原文。
2. 允许基于知识库做合理逻辑推理（如初中生→学生）。
3. 知识库没有的具体信息（年龄、地址等），明确说"知识库中没有相关信息"。
4. 回答简洁自然，不要复述规则。
"""


def build_chat_prompt(user_input: str, short_history: list, tool_info: str) -> str:
    """闲聊模式：放开约束，允许模型用自身常识回答"""
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    return f"""
【用户提问】{user_input}

【近期对话记录】
{history_str}

【工具查询结果】
{tool_info if tool_info else "无"}

【回答规则】
1. 结合工具查询结果和你的常识，自然地回答用户问题。
2. 如果工具结果中有相关信息，优先参考工具结果。
3. 回答简洁友好，像正常聊天一样。
"""


def main():
    print("=== Demo14 意图分类 + 多工具智能体 ===")
    sources = list_knowledge_sources()
    print(f"已加载知识库：{sources if sources else '（空）'}")
    print("支持：时间查询 / 数学计算 / 联网搜索 / 知识库问答")
    print("输入exit退出对话\n")

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

        # 第四步：根据意图选择不同流程
        if intent == "knowledge":
            # 知识库模式：先检索，检索为空则转闲聊
            rag_text = search_knowledge(user_input, top_n=3)
            if rag_text == "暂无相关知识库内容":
                # 检索为空 → 转闲聊模式，放开模型回答
                print("[知识库无匹配，切换闲聊模式]")
                full_prompt = build_chat_prompt(user_input, short_history, tool_info)
            else:
                # 有检索结果 → 严格知识库模式
                full_prompt = build_knowledge_prompt(user_input, short_history, tool_info, rag_text)
        else:
            # 工具类意图（时间/计算/搜索）：用闲聊模式，工具结果已经包含答案
            full_prompt = build_chat_prompt(user_input, short_history, tool_info)

        # 第五步：模型生成回答
        response = agent.run(full_prompt)
        print(f"Agent：{response}\n")

        # 第六步：保存对话
        append_chat_message("user", user_input)
        append_chat_message("assistant", response)


if __name__ == "__main__":
    main()
