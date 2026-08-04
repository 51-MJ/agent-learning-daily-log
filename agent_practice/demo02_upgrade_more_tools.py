from smolagents import LiteLLMModel, CodeAgent, tool
import datetime
# 1.连接本地ollama模型
# ollama默认地址 http://localhost:11434
model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="dummy"
)

# 2.创建工具
# 工具一：获取当前时间
@tool
def get_now_time() -> str:
    """获取电脑当前时间 """
    now = datetime.datetime.now()
    return now.strftime("%H-%M-%S")

# 工具二：简易计算器
@tool
def get_today_date() ->str:
    """获取电脑当前日期,包括年月日，星期几"""
    week_map = {0:"星期一", 1:"星期二", 2:"星期三", 3:"星期四", 4:"星期五", 5:"星期六", 6:"星期日"}
    now = datetime.datetime.now()
    week = week_map[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日 {week}"

# 工具三：升级版四则运算器
#@tool
def calc_four_operation(a: float, b: float, op: str) -> float:
    """
    执行加减乘除四则运算，调用此工具必须传入两个数字和一个运算符
    Args:
        a: 参与运算的第一个数字，数字类型
        b: 参与运算的第二个数字，数字类型
        op: 运算符号，只能是字符串 "+"、"-"、"*"、"/" 其中一个
    """
    if op == "+":
        res=a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    elif op == "/":
        
        if b == 0:
            return "除数不能为0"
        res = a / b
    else:
        return "不支持的运算符，请使用 + - * /"
    return f"计算结果：{res}"

# 工具四： 文本字数统计
#@tool
def count_text_length(content: str) -> str:
    """
    统计一段文本的字符数量
    Args:
        content: 需要统计字数的文本内容
    """
    length = len(content)
    return f"文本长度为：{length} 个字符"

# 3.初始化Agent，注册可用工具
agent= CodeAgent(
    model=model,
    tools=[get_now_time,get_today_date],
    add_base_tools=False
)

# 4.测试提问
if __name__ == "__main__":
    test_question = [
        "现在几点了？",
        "今天是几号？",
        "今天是星期几？",
        # "abcdfdf情书一下这段话有多少字？"
    ]
    for question in test_question:
        print("用户提问:", question)
        response = agent.run(question)
        print("AI回答:", response)