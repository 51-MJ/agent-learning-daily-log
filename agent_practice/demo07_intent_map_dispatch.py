"""
Demo07：意图字典映射调度工具
所属阶段：阶段4 Agent适配优化
复用 memory_utils.py 做对话持久化
功能：支持多意图同时触发，字典替代if‑elif分支
"""
import sys
from pathlib import Path
# 把上一级目录加入python搜索路径
sys.path.append(str(Path(__file__).parent.parent))
from memory_utils import append_chat_message
from smolagents import CodeAgent, LiteLLMModel, tool
from datetime import datetime

@tool
def get_now_time() -> str:
    """获取电脑当前时间"""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

@tool
def get_today_date() -> str:
    """获取电脑当前日期，包含年月日和星期几"""
    week_map = {0:"星期一",1:"星期二",2:"星期三",3:"星期四",4:"星期五",5:"星期六",6:"星期日"}
    now = datetime.now()
    week = week_map[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日 {week}"

#模型初始化
model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://127.0.0.1:11434"
)
agent = CodeAgent(
    tools = [get_now_time,get_today_date],
    model = model
)

#意图映射表
#key:意图标签，value:对应的工具函数
intent_handler_map = {
    "query_time": get_now_time,
    "query_date": get_today_date,
}

def intent_classify(user_input: str) -> list:
    """简单关键词意图分类，返回意图标签列表，支持多意图"""
    tags = []
    if "时间" in user_input or "几点" in user_input:
        tags.append("query_time")
    if "日期" in user_input or "几号" in user_input or "星期几" in user_input:
        tags.append("query_date")
    return tags

# ===================== 主循环 =====================
if __name__ == "__main__":
    print("==== Demo07 启动 | 输入 exit 退出 ====")                          
    while True:
        user_msg = input("\n请输入你的问题（输入exit退出）：")
        if user_msg.strip() == "exit":
            break

        intent_tags = intent_classify(user_msg)
        tool_results = []

        # 遍历所有识别出来的意图，查表执行工具
        for tag in intent_tags:
            handle_func = intent_handler_map.get(tag)
            if handle_func is not None:
                res = handle_func()
                tool_results.append(res)

        # 有工具结果就拼接输出，没有交给agent闲聊
        if len(tool_results) > 0:
             # 将工具原始结果连同用户原始提问，一起交给Agent整合回答，兼顾工具数据与闲聊内容
            combined_prompt = f"""用户完整提问：{user_msg}
                                工具查询得到的参考数据：{"\n".join(tool_results)}

                                严格遵守以下回答规则：
                                1. 用户提问包含多个问题、闲聊、查询需求时，**必须全部完整回应，禁止只回答工具相关内容**；
                                2. 若用户询问工具名称、工具来源、调用记录，优先回答工具相关问题，再处理时间；
                                3. 闲聊、经济/金融话题需要正常展开交流，不能跳过；
                                4. 工具数据仅作为参考，不要只围绕时间数据输出。
                                """
            answer = agent.run(combined_prompt)
        else:
            # 没有匹配到工具意图，直接正常对话
            answer = agent.run(user_msg)

        print(f"agent > {answer}\n")
        # 调用公共模块保存一轮对话
        append_chat_message("user", user_msg)
        append_chat_message("assistant", answer)