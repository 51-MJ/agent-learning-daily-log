# demo09_rag_agent.py
# 功能：标准强制全局RAG智能体，任意提问自动检索知识库
import sys
from pathlib import Path
# 将项目根目录加入python路径，解决utils导入失败
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from smolagents import LiteLLMModel, CodeAgent
from memory_utils import load_chat_history, append_chat_message, get_history_count
from tool_router import run_time_tools, search_knowledge

# 初始化本地Qwen2.5-7b模型
model = LiteLLMModel(model_id="ollama/qwen2.5:7b", temperature=0.7)
agent = CodeAgent(model=model, tools=[])

def build_prompt(user_text: str, time_info: str, rag_content: str, history_list: list):
    """组装提示词：对话历史 + 时间工具数据 + 强制检索知识库片段"""
    recall_str = ""
    for idx, item in enumerate(history_list, 1):
        recall_str += f"{idx}. {item['role']}：{item['content']}\n"
    
    base_prompt = f"""
【硬性输出要求：回答第一行固定输出【用户提问：{user_text}】】
【近期对话记录】
{recall_str}
【时间工具查询结果】
{time_info}
【本地知识库检索片段（每次提问自动查询，优先参考）】
{rag_content}

硬性回答规则（必须全部遵守，不可违背）：
1. 优先阅读知识库片段，只要片段包含和用户问题相关内容，必须引用知识库内容作答；
2. 禁止仅依靠历史对话就判定无信息，必须优先参考知识库；
3. 用户同时询问时间+文档内容时，先输出完整时间信息，再结合知识库资料作答；
4. 只有知识库完全空白、无任何相关内容时，才能仅根据对话历史闲聊回复；
5. 语言简洁自然，逻辑清晰，无冗余重复。

用户当前提问：{user_text}
"""
    return base_prompt

if __name__ == "__main__":
    print("==== Demo09 标准强制RAG智能体 | 输入exit退出对话 ====")
    while True:
        user_msg = input("\n用户：")
        if user_msg.strip().lower() == "exit":
            print("对话结束，所有对话已自动保存至chat_history.json")
            break
        
        # 1. 读取历史对话
        chat_history = load_chat_history()
        print(f"当前加载短期对话条数：{get_history_count()}")

        # 2. 关键词匹配执行时间工具（可选执行）
        time_result = run_time_tools(user_msg)
        # 3. 【标准强制RAG核心】任意提问都自动检索知识库，不受关键词限制
        rag_result = search_knowledge(user_msg)
        
        # 4. 组装完整提示词送入模型
        full_prompt = build_prompt(user_msg, time_result, rag_result, chat_history)
        answer = agent.run(full_prompt)
        
        # 输出模型回答
        print(f"Agent：{answer}")
        
        # 5. 保存本轮对话记录
        append_chat_message("user", user_msg)
        append_chat_message("assistant", answer)