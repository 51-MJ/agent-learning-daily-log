# demo05 功能：意图识别 + 结合持久化记忆应答
# 学习目标：
# 1. 理解用户意图的概念，区分用户提问的不同目的
# 2. 通过关键词/模型识别意图，精准匹配对应工具或调取历史对话
# 3. 复用demo04的chat_history.json持久化对话数据
# 4. 解决小模型自主调用工具不稳定的问题，人为约束决策流程
from  smolagents import  LiteLLMModel, CodeAgent,tool
import datetime
import json
import os

modle = LiteLLMModel(
    model_id = "ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="dummy"
)

@tool
def get_now_time() ->str:
    """获取电脑当前时间
    returns:
        str: 当前时间，格式为 HH-MM-SS
    """
    now = datetime.datetime.now()
    return now.strftime("%H-%M-%S")

@tool
def get_today_date() ->str:
    """获取电脑当前日期，包括年月日，星期几
    returns:
        str: 当前日期，格式为 YYYY年MM月DD日 星期X
    """
    week_map = {0:"星期一", 1:"星期二", 2:"星期三", 3:"星期四", 4:"星期五", 5:"星期六", 6:"星期日"}
    now = datetime.datetime.now()
    week = week_map[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日 {week}"

HISTORY_FILE = "chat_history.json"    
chat_history = []

# 加载历史对话
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf_8") as f:
        chat_history = json.load(f)


## 创建智能体
agent = CodeAgent(
    model=modle,
    tools=[get_now_time,get_today_date],
    add_base_tools=False
)

#核心：意图识别函数（通过关键词判断用户意图）
def detect_intent(user_text: str) -> str:
    """
    识别用户输入对应的意图标签
    :param user_text: 用户输入的提问文本
    :return: 意图标签 query_time / query_date / recall_history / normal_chat
    """

    text = user_text.lower()
    matched_intents = []
    #查询时间意图
    if any(word in text for word in ["时间","几点","几点了","现在时间"]):
        matched_intents.append("query_time")
    #查询日期意图
    if any(word in text for word in ["日期","今天几号","星期几","今天星期几"]):
        matched_intents.append("query_date")
    #回忆历史对话意图
    if any(word in text for word in ["之前","记得","上次","之前说过""聊了什么"]):
        matched_intents.append("recall_history")
    #去重
    matched_intents = list(set(matched_intents))

    if len(matched_intents) == 0:
        matched_intents.append("normal_chat")
    return matched_intents


if __name__ == "__main__":
    print("===== demo05【多意图升级版】意图识别+长期记忆智能体启动 =====")
    print(f"已加载历史对话条数：{len(chat_history)}")
    print("输入 exit 即可退出程序\n")

    while True:
        user_msg = input("你：")
        if user_msg.lower() == "exit":
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)
            print("助手：对话已持久化保存，本次会话结束。")
            break

        intent_list = detect_intent(user_msg)
        answer_parts = []

        # 根据意图调用不同工具或逻辑
        if "query_time" in intent_list:
            res_time = agent.run("帮我获取当前时间")
            answer_parts.append(str(res_time))
        if "query_date" in intent_list:
            res_date = agent.run("帮我获取今天的日期和星期")
            answer_parts.append(str(res_date))
        if "recall_history" in intent_list:
            if len(chat_history) == 0:
                answer_parts.append(str("我没有记忆到之前的对话内容。"))
            else:
                recall_text = "过往聊天记录如下：\n"
                for idx,item in enumerate(chat_history,1):
                    role_name = "用户" if item["role"] == "user" else "智能体"
                    recall_text += f"{idx}. {role_name}：{item['content']}\n"
                answer_parts.append(recall_text)

        if "normal_chat" in intent_list :
            answer_parts.append(agent.run(user_msg))

         # 拼接所有结果生成最终回答
        answer = "\n".join(answer_parts)

        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": answer})
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)