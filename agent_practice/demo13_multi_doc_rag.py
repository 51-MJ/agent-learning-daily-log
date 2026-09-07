# demo13_multi_doc_rag.py
# 功能：多文档知识库 + 回答来源标注 + 增量更新支持
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from smolagents import CodeAgent, LiteLLMModel
from memory_utils import append_chat_message, slide_window_history
from tool_router import run_time_tools
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


def build_prompt(user_input: str, short_history: list, tool_info: str, rag_text: str) -> str:
    """组装带来源标注要求的提示词"""
    history_str = ""
    for idx, item in enumerate(short_history, 1):
        history_str += f"{idx}. {item['role']}：{item['content']}\n"

    prompt = f"""
【近期对话记录（仅用于理解上下文，不可作为事实依据）】
{history_str}

【时间工具查询结果】
{tool_info}

【本地知识库检索内容（回答的唯一事实来源，每个片段标注了来源文件）】
{rag_text}

【用户当前提问】
{user_input}

【硬性回答规则（必须全部严格遵守）】
1. 所有事实性信息必须来自上方【本地知识库检索内容】，优先引用知识库原文作答。
2. 允许基于知识库内容做合理的逻辑推理（例如：知识库说"初中生"，可推理出"职业是学生"），但推理必须逻辑必然、无歧义。
3. 严格禁止编造知识库中不存在的具体事实，包括但不限于：年龄、出生日期、学校名称、家庭住址、电话号码、具体爱好、具体收入、具体经历等。知识库没写的信息，一律说"知识库中没有相关信息"。
4. 当知识库检索内容为"暂无相关知识库内容"时，直接回复："未查询到相关信息，知识库中没有关于这个问题的内容。"，禁止依靠模型自身常识回答。
5. 对话历史仅用于理解上下文语境，若历史对话内容与知识库冲突，以知识库为准。
6. 【来源标注要求】回答时必须区分信息来源：
   - 【知识库原文】：知识库中直接包含的信息，和用户问题语义一致，不需要推导，直接引用即可。
     正例：知识库说"不关注理财投资"，用户问"关注理财吗"，回答"不关注"→【知识库原文】
     正例：知识库说"马艳是初中生"，用户问"马艳是初中生吗"，回答"是"→【知识库原文】
   - 【逻辑推理】：基于知识库内容做进一步推导才能得出的结论。
     正例：知识库说"马艳是初中生"，用户问"马艳的职业是什么"，回答"学生"→【逻辑推理】
   - 【无依据】：知识库中完全没有相关信息，直接说明"知识库中没有相关信息"。

7. 回答简洁自然，不要复述规则本身，不要输出多余的解释。
"""
    return prompt


def main():
    print("=== Demo13 多文档知识库 + 来源标注智能体 ===")
    # 启动时显示当前加载的知识库来源
    sources = list_knowledge_sources()
    print(f"已加载知识库来源：{sources if sources else '（空，请先运行build_knowledge_base.py）'}")
    print("输入exit退出对话\n")

    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break

        short_history = slide_window_history(max_turn=6)
        tool_info = run_time_tools(user_input)
        rag_text = search_knowledge(user_input, top_n=3)

        full_prompt = build_prompt(user_input, short_history, tool_info, rag_text)
        response = agent.run(full_prompt)
        print(f"Agent：{response}\n")

        append_chat_message("user", user_input)
        append_chat_message("assistant", response)


if __name__ == "__main__":
    main()
