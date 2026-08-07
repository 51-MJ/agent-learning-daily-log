# demo08_tool_router.py
# 功能：主对话程序，复用记忆模块 + 调用tool_router实现多工具批量调度
import sys
from pathlib import Path
# 将项目根目录加入python路径，解决同级根目录utils文件导入失败
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from smolagents import LiteLLMModel,  CodeAgent
from memory_utils import load_chat_history, append_chat_message, get_history_count
from tool_router import run_all_matched_tools

# 初始化本地Qwen2.5-7b模型
model = LiteLLMModel(model_id="ollama/qwen2.5:7b", temperature=0.7)
agent = CodeAgent(model=model, tools=[])

def build_prompt(user_text: str, tool_info: str, history_list: list):
    """组装完整提示词：历史对话 + 工具返回数据 + 用户提问"""
    recall_str = ""
    for idx, item in enumerate(history_list, 1):
        recall_str += f"{idx}. {item['role']}：{item['content']}\n"
    
    base_prompt = f"""
【历史对话记录】
{recall_str}
【工具实时查询结果】{tool_info}

硬性回答规则（必须全部遵守）：
1. 如果工具实时查询结果不为空，你的回答**必须完整包含所有工具给出的时间信息，不能省略、不能只聊闲聊**；
2. 用户同时提问时间和金融话题时，先给出完整时间信息，再回答金融相关内容；
3. 仅当工具结果为空时，只回应闲聊内容；
4. 语言自然简洁，不要重复冗余文字。

用户当前提问：{user_text}
"""
    return base_prompt

if __name__ == "__main__":
    print("==== Demo08 多工具路由智能体（输入exit退出）====")
    while True:
        user_msg = input("\n用户：")
        if user_msg.strip().lower() == "exit":
            print("对话结束，历史已保存")
            break
        
        # 1. 读取滑动窗口截断后的历史对话
        chat_history = load_chat_history()
        print(f"当前加载历史对话条数：{get_history_count()}")
        
        # 2. 路由批量匹配、执行所有工具
        tool_result = run_all_matched_tools(user_msg)
        
        # 3. 拼接prompt送入模型生成回答
        full_prompt = build_prompt(user_msg, tool_result, chat_history)
        answer = agent.run(full_prompt)
        
        # 4. 打印回复
        print(f"Agent：{answer}")
        
        # 5. 保存本轮对话至json（标准格式，无参数颠倒bug）
        append_chat_message("user", user_msg)
        append_chat_message("assistant", answer)