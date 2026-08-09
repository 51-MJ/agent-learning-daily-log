# demo11_full_rag_optimize.py
# 功能：持久化向量库+重叠分片RAG优化，搭配滑动窗口记忆、工具路由
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from smolagents import CodeAgent, LiteLLMModel
from memory_utils import append_chat_message, slide_window_history
from tool_router import run_time_tools
from rag_utils import search_knowledge

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

def main():
    print("=== Demo11 持久化分片优化RAG智能体 ===")
    print("输入exit退出对话\n")
    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break
        
        # 滑动窗口读取短期对话上下文
        short_history = slide_window_history(max_turn=6)
        # 时间工具调度
        tool_info = run_time_tools(user_input)
        # 优化后的RAG检索
        rag_text = search_knowledge(user_input, top_n=3)

        prompt = f"""
近期对话记录（最多保留6轮）：{short_history}
时间工具查询结果：{tool_info}
知识库参考内容：{rag_text}
用户当前最新提问：{user_input}
硬性回答规则（必须严格遵守）：
1. 所有回答优先以知识库内容为唯一权威依据；
2. 若历史对话内容与知识库信息发生冲突，直接忽略对话历史，完全按照知识库内容作答；
3. 不要复述、重复上一轮无关对话答案，只针对本次提问作答；
4. 只有知识库无相关内容时，才可参考对话历史闲聊回复。
"""
        response = agent.run(prompt)
        print(f"Agent：{response}\n")

        # 完整对话永久保存
        append_chat_message("user", user_input)
        append_chat_message("assistant", response)

if __name__ == "__main__":
    main()