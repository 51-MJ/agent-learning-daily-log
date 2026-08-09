# demo10_slide_window_rag.py
# 功能：新增对话滑动窗口，限制历史长度，解决长对话上下文过长、推理耗时增加问题
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from smolagents import CodeAgent, tool, LiteLLMModel
# 导入新增的滑动窗口函数
from memory_utils import load_chat_history, append_chat_message, slide_window_history
from tool_router import run_time_tools
from rag_utils import search_knowledge

# 模型配置（沿用原有7B模型，不更换轻量化小模型）
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
    print("=== Demo10 滑动窗口优化RAG智能体 ===")
    print("输入exit退出对话\n")
    while True:
        user_input = input("用户：")
        if user_input.strip().lower() == "exit":
            print("对话结束")
            break
        
        # 核心改动：不直接读取完整历史，调用滑动窗口截断
        chat_history = slide_window_history(max_turn=6)
        
        # 2. 时间工具匹配、执行全部交给tool_router，主程序无关键词判断逻辑
        tool_info = run_time_tools(user_input)
        
        # 3. 全局知识库检索
        rag_text = search_knowledge(user_input)

        
        # 组装提示词
        prompt = f"""
    近期对话记录（最多保留6轮）：{chat_history}
    时间工具查询结果：{tool_info}
    知识库参考内容：{rag_text}
    用户当前最新提问：{user_input}
    硬性要求：
    1. 结合知识库、工具结果完整回答当前用户最新问题；
    2. 不要复述、重复上一轮对话的答案，只针对本次提问作答；
    3. 不要忽略用户闲聊类诉求。
    """
        res = agent.run(prompt)
        print(f"Agent：{res}\n")
        
        # 完整对话依旧存入json（滑动窗口仅读取时截断，原始记录完整保存）
        append_chat_message("user", user_input)
        append_chat_message("assistant", res)

if __name__ == "__main__":
    main()