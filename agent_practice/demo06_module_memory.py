from smolagents import LiteLLMModel, CodeAgent, tool
import datetime
# 导入我们刚写的公共记忆模块
import sys
# 让脚本可以找到项目根目录，能导入memory_utils
sys.path.append(r"D:\develop\agent-learing\agent-learning-daily-log")
from memory_utils import load_chat_history, append_chat_message, get_history_count

# 1.连接本地ollama qwen模型
model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="dummy"
)

# 2.定义工具
@tool
def get_now_time() -> str:
    """获取电脑当前时间"""
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

@tool
def get_today_date() -> str:
    """获取电脑当前日期，包含年月日和星期几"""
    week_map = {0:"星期一",1:"星期二",2:"星期三",3:"星期四",4:"星期五",5:"星期六",6:"星期日"}
    now = datetime.datetime.now()
    week = week_map[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日 {week}"

# 3.创建智能体
agent = CodeAgent(
    tools=[get_now_time, get_today_date],
    model=model,
    add_base_tools=False
)

# 4.意图识别函数
def detect_intent(user_input: str):
    intent_list = []
    lower_text = user_input.lower()
    if "几点" in lower_text:
        intent_list.append("query_time")
    if "星期几" in lower_text or "几号" in lower_text:
        intent_list.append("query_date")
    if ("之前聊了什么" in lower_text) or ("我之前问了你什么" in lower_text) or ("刚才问了你什么" in lower_text):
        intent_list.append("recall_history")
    # 只要不是纯工具查询，就附带闲聊意图
    intent_list.append("normal_chat")
    # 去重
    intent_list = list(set(intent_list))
    return intent_list

# 启动加载历史
chat_history = load_chat_history()
print(f"已加载历史对话条数：{get_history_count()}")

# 交互循环
if __name__ == "__main__":
    while True:
        user_q = input("\n请输入你的问题（输入exit退出）：")
        if user_q == "exit":
            break
        # 识别意图
        intents = detect_intent(user_q)
        answer_parts = []

         # 1.回忆历史：本地直接读json，不耗模型
        if "recall_history" in intents:
            recall_text = "过往所有对话记录：\n"
            history_data = load_chat_history()
            for idx, msg in enumerate(history_data, 1):
                recall_text += f"{idx}. {msg['role']}：{msg['content']}\n"
            answer_parts.append(recall_text)

        # 2.关键：把识别出的意图一起传给模型，只跑一次agent.run
        prompt = f"""用户原始问题：{user_q}
                    已经识别出用户意图标签：{intents}
                    请你结合意图回答：
                    - 如果包含query_time，务必调用get_now_time工具获取当前时间
                    - 如果包含query_date，务必调用get_today_date工具获取日期星期
                    - 如果包含normal_chat，请正常和用户闲聊互动
                """
        model_answer = agent.run(prompt)
        answer_parts.append(str(model_answer))

        final_answer = "\n".join(answer_parts)
        print("智能体回答：\n", final_answer)
        # 调用公共函数自动保存本轮对话
        append_chat_message("user", user_q)
        append_chat_message("assistant", final_answer)